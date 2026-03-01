# Instruction Set Architecture (ISA)

**Previous**: [08_Control_Unit.md](./08_Control_Unit.md) | **Next**: [10_Assembly_Language_Basics.md](./10_Assembly_Language_Basics.md)

**Difficulty**: ⭐⭐⭐

**Prerequisites**: Basic CPU structure, control unit, binary representation

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define ISA as the contract between hardware and software
2. Classify instruction types (data transfer, arithmetic, control flow)
3. Compare RISC and CISC design philosophies with concrete examples
4. Explain the major addressing modes (immediate, register, direct, indirect, indexed)
5. Analyze how ISA choices affect compiler design and program performance
6. Describe the MIPS/ARM instruction format as a RISC example

---

The Instruction Set Architecture is the most important interface in computing -- it is the contract between software and hardware. Every program you write, no matter the language, is ultimately compiled down to ISA instructions. The RISC vs. CISC debate shaped the entire modern processor landscape, from ARM chips in your phone to x86 in your laptop.

## Table of Contents

1. [ISA Concepts](#1-isa-concepts)
2. [CISC vs RISC Comparison](#2-cisc-vs-risc-comparison)
3. [Instruction Formats](#3-instruction-formats)
4. [Addressing Modes](#4-addressing-modes)
5. [Major ISAs](#5-major-isas)
6. [Practice Problems](#6-practice-problems)

---

> Think of the ISA as a contract or API between hardware and software. The hardware promises to implement a specific set of operations (instructions), and the software (compilers, OS) promises to use only those operations. Neither side needs to know how the other works internally -- this abstraction is what lets you run the same program on different CPU implementations.

## 1. ISA Concepts

### 1.1 What is an ISA?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ISA: The Contract Between Software and Hardware       │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                        Software                                  │ │
│    │                                                                 │ │
│    │    ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │ │
│    │    │ Applications │    │  Compiler   │    │ Operating System│   │ │
│    │    │  (C, Java)  │    │   (GCC)     │    │  (Linux, Win)   │   │ │
│    │    └─────────────┘    └─────────────┘    └─────────────────┘   │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                   │                                     │
│                                   │ Abstraction Layer                   │
│                                   ▼                                     │
│    ╔═════════════════════════════════════════════════════════════════╗ │
│    ║              ISA (Instruction Set Architecture)                 ║ │
│    ║                                                                 ║ │
│    ║    - Instructions                                               ║ │
│    ║    - Registers                                                  ║ │
│    ║    - Data Types                                                 ║ │
│    ║    - Addressing Modes                                           ║ │
│    ║    - Memory Model                                               ║ │
│    ║    - I/O                                                        ║ │
│    ║    - Exception Handling                                         ║ │
│    ╚═════════════════════════════════════════════════════════════════╝ │
│                                   │                                     │
│                                   │ Implementation                      │
│                                   ▼                                     │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                         Hardware                                 │ │
│    │                                                                 │ │
│    │    ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │ │
│    │    │    CPU      │    │   Cache     │    │     Memory      │   │ │
│    │    │  Micro-     │    │            │    │                  │   │ │
│    │    │ architecture│    │            │    │                  │   │ │
│    │    └─────────────┘    └─────────────┘    └─────────────────┘   │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 What an ISA Defines

| Component | Description | Examples |
|-----------|-------------|----------|
| Instruction Set | Operations the processor can execute | ADD, SUB, LOAD, STORE, JUMP |
| Registers | Programmer-accessible registers | x86: EAX, EBX / ARM: R0-R15 |
| Data Types | Supported data formats | byte, word, integer, floating-point |
| Instruction Format | Bit encoding of instructions | R-type, I-type, J-type |
| Addressing Modes | How to specify operand locations | immediate, direct, indirect, register |
| Memory Model | Memory access methods and alignment rules | Little/Big Endian, alignment requirements |
| Exceptions/Interrupts | How to handle exceptional conditions | traps, interrupt vectors |

### 1.3 ISA vs Microarchitecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Relationship Between ISA and Microarchitecture            │
│                                                                         │
│    One ISA ─────────────────────────────────────────┐                   │
│         │                                          │                    │
│         │   Can be implemented by multiple         │                    │
│         │   microarchitectures                     │                    │
│         │                                          │                    │
│         ▼                                          ▼                    │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
│    │ x86 ISA      │    │ x86 ISA      │    │ x86 ISA      │            │
│    │              │    │              │    │              │            │
│    │ Intel Core   │    │ Intel Atom   │    │ AMD Zen 3    │            │
│    │(High-perf)   │    │(Low-power)   │    │(Competitor)  │            │
│    └──────────────┘    └──────────────┘    └──────────────┘            │
│                                                                         │
│    The same program runs on all implementations!                        │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                          Analogy                                │ │
│    │                                                                 │ │
│    │    ISA = Car driving interface (steering, pedals, gears)        │ │
│    │    Microarchitecture = Engine implementation (gas, diesel, EV)  │ │
│    │                                                                 │ │
│    │    → Driver operates the same way regardless of engine type     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CISC vs RISC Comparison

### 2.1 CISC (Complex Instruction Set Computer)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CISC Characteristics                           │
│                                                                         │
│    Philosophy: "Perform complex operations with a single instruction"   │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                    CISC Instruction Example (x86)               │ │
│    │                                                                 │ │
│    │    REP MOVSB     ; Memory block copy (repeat + move)            │ │
│    │                                                                 │ │
│    │    What this single instruction does:                           │ │
│    │    1. Check ECX register value (repeat count)                   │ │
│    │    2. Read byte from ESI                                        │ │
│    │    3. Write byte to EDI                                         │ │
│    │    4. Increment/decrement ESI, EDI                              │ │
│    │    5. Decrement ECX                                             │ │
│    │    6. Repeat if ECX > 0                                         │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Characteristics:                                                     │
│    ┌───────────────────┬────────────────────────────────────────────┐  │
│    │ Number of instr.  │ Many (hundreds to thousands)               │  │
│    ├───────────────────┼────────────────────────────────────────────┤  │
│    │ Instruction length│ Variable (1 - 15 bytes)                    │  │
│    ├───────────────────┼────────────────────────────────────────────┤  │
│    │ Addressing modes  │ Diverse (12+ modes)                        │  │
│    ├───────────────────┼────────────────────────────────────────────┤  │
│    │ Memory access     │ Direct memory reference in instructions    │  │
│    ├───────────────────┼────────────────────────────────────────────┤  │
│    │ Execution cycles  │ Varies per instruction (1 - tens)          │  │
│    ├───────────────────┼────────────────────────────────────────────┤  │
│    │ Control unit      │ Microprogrammed (mainly)                   │  │
│    └───────────────────┴────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 RISC (Reduced Instruction Set Computer)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RISC Characteristics                           │
│                                                                         │
│    Philosophy: "Execute simple instructions quickly"                    │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                    RISC Instruction Example (MIPS)              │ │
│    │                                                                 │ │
│    │    Instruction sequence for memory copy:                        │ │
│    │                                                                 │ │
│    │    loop:                                                        │ │
│    │        lb   $t0, 0($s0)      ; Load byte from source            │ │
│    │        sb   $t0, 0($s1)      ; Store byte to destination        │ │
│    │        addi $s0, $s0, 1      ; Increment source pointer         │ │
│    │        addi $s1, $s1, 1      ; Increment destination pointer    │ │
│    │        addi $t1, $t1, -1     ; Decrement counter                │ │
│    │        bne  $t1, $zero, loop ; Repeat if not zero               │ │
│    │                                                                 │ │
│    │    Decomposed into 6 simple instructions                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Characteristics:                                                     │
│    ┌───────────────────┬────────────────────────────────────────────┐  │
│    │ Number of instr.  │ Few (tens to ~100)                         │  │
│    ├───────────────────┼────────────────────────────────────────────┤  │
│    │ Instruction length│ Fixed (32 bits)                            │  │
│    ├───────────────────┼────────────────────────────────────────────┤  │
│    │ Addressing modes  │ Limited (3-5 modes)                        │  │
│    ├───────────────────┼────────────────────────────────────────────┤  │
│    │ Memory access     │ Load/Store only (arithmetic on registers)  │  │
│    ├───────────────────┼────────────────────────────────────────────┤  │
│    │ Execution cycles  │ Mostly 1 cycle (pipelining)                │  │
│    ├───────────────────┼────────────────────────────────────────────┤  │
│    │ Control unit      │ Hardwired                                  │  │
│    └───────────────────┴────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Detailed CISC vs RISC Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CISC vs RISC Comparison Table                       │
├──────────────────────┬─────────────────────┬────────────────────────────┤
│        Aspect        │       CISC          │          RISC              │
├──────────────────────┼─────────────────────┼────────────────────────────┤
│ Representative ISA   │ x86, x86-64         │ ARM, MIPS, RISC-V          │
├──────────────────────┼─────────────────────┼────────────────────────────┤
│ Instruction format   │ Variable length     │ Fixed length               │
│                      │ (1-15 bytes)        │ (4 bytes)                  │
├──────────────────────┼─────────────────────┼────────────────────────────┤
│ Number of registers  │ Few (8-16)          │ Many (32+)                 │
├──────────────────────┼─────────────────────┼────────────────────────────┤
│ Memory operations    │ All instructions    │ Load/Store only            │
├──────────────────────┼─────────────────────┼────────────────────────────┤
│ Compiler complexity  │ Low                 │ High                       │
├──────────────────────┼─────────────────────┼────────────────────────────┤
│ Hardware complexity  │ High                │ Low                        │
├──────────────────────┼─────────────────────┼────────────────────────────┤
│ Pipelining           │ Difficult           │ Easy                       │
├──────────────────────┼─────────────────────┼────────────────────────────┤
│ Code density         │ High                │ Low                        │
├──────────────────────┼─────────────────────┼────────────────────────────┤
│ Power efficiency     │ Low                 │ High                       │
├──────────────────────┼─────────────────────┼────────────────────────────┤
│ Primary use cases    │ Desktop, Server     │ Mobile, Embedded           │
└──────────────────────┴─────────────────────┴────────────────────────────┘
```

### 2.4 Modern Perspective

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Convergence in Modern Processors                     │
│                                                                         │
│    Modern x86 Processors (Intel/AMD):                                   │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                                                                 │ │
│    │         x86 CISC Instructions                                   │ │
│    │              │                                                  │ │
│    │              ▼                                                  │ │
│    │    ┌─────────────────────┐                                      │ │
│    │    │  Instruction Decoder│                                      │ │
│    │    │  (CISC → micro-ops) │                                      │ │
│    │    └──────────┬──────────┘                                      │ │
│    │               │                                                 │ │
│    │               ▼                                                 │ │
│    │    ┌─────────────────────┐                                      │ │
│    │    │  RISC-style Core    │                                      │ │
│    │    │  (executes micro-ops)│                                     │ │
│    │    │  - Pipelining       │                                      │ │
│    │    │  - Superscalar      │                                      │ │
│    │    │  - Out-of-order     │                                      │ │
│    │    └─────────────────────┘                                      │ │
│    │                                                                 │ │
│    │    Conclusion: CISC on the outside, RISC on the inside!         │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ARM Processors:                                                      │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │    - Maintains basic RISC design                                │ │
│    │    - Added some complex instructions (SIMD, encryption, etc.)   │ │
│    │    - Entering desktop/server market (Apple M4, AWS Graviton4)    │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Instruction Formats

### 3.1 MIPS Instruction Formats

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MIPS Instruction Formats (32-bit)                    │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                        R-type (Register)                        │ │
│    │                                                                 │ │
│    │   31    26 25   21 20   16 15   11 10    6 5      0            │ │
│    │   ┌──────┬───────┬───────┬───────┬───────┬────────┐            │ │
│    │   │opcode│  rs   │  rt   │  rd   │ shamt │ funct  │            │ │
│    │   │6 bits│5 bits │5 bits │5 bits │5 bits │6 bits  │            │ │
│    │   └──────┴───────┴───────┴───────┴───────┴────────┘            │ │
│    │                                                                 │ │
│    │   Example: ADD $rd, $rs, $rt                                    │ │
│    │       opcode=0, funct=0x20 (add)                                │ │
│    │       rd = rs + rt                                              │ │
│    │                                                                 │ │
│    │   Example: SLL $rd, $rt, shamt                                  │ │
│    │       rd = rt << shamt                                          │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                        I-type (Immediate)                       │ │
│    │                                                                 │ │
│    │   31    26 25   21 20   16 15                  0               │ │
│    │   ┌──────┬───────┬───────┬─────────────────────┐               │ │
│    │   │opcode│  rs   │  rt   │    immediate        │               │ │
│    │   │6 bits│5 bits │5 bits │     16 bits         │               │ │
│    │   └──────┴───────┴───────┴─────────────────────┘               │ │
│    │                                                                 │ │
│    │   Example: ADDI $rt, $rs, imm                                   │ │
│    │       rt = rs + sign_extend(imm)                                │ │
│    │                                                                 │ │
│    │   Example: LW $rt, offset($rs)                                  │ │
│    │       rt = Memory[rs + sign_extend(offset)]                     │ │
│    │                                                                 │ │
│    │   Example: BEQ $rs, $rt, offset                                 │ │
│    │       if (rs == rt) PC = PC + 4 + offset * 4                    │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                        J-type (Jump)                            │ │
│    │                                                                 │ │
│    │   31    26 25                                 0                │ │
│    │   ┌──────┬─────────────────────────────────────┐               │ │
│    │   │opcode│              address                │               │ │
│    │   │6 bits│              26 bits                │               │ │
│    │   └──────┴─────────────────────────────────────┘               │ │
│    │                                                                 │ │
│    │   Example: J target                                             │ │
│    │       PC = (PC[31:28] << 28) | (address << 2)                   │ │
│    │                                                                 │ │
│    │   Example: JAL target                                           │ │
│    │       $ra = PC + 4; PC = target                                │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 MIPS Instruction Encoding Examples

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   MIPS Instruction Encoding Examples                     │
│                                                                         │
│    Example 1: ADD $t0, $s1, $s2                                         │
│    ────────────────────────────────────────────────                     │
│    R-type: rd = rs + rt                                                 │
│    $t0 = 8, $s1 = 17, $s2 = 18                                          │
│                                                                         │
│    ┌──────┬───────┬───────┬───────┬───────┬────────┐                   │
│    │000000│ 10001 │ 10010 │ 01000 │ 00000 │ 100000 │                   │
│    │  op  │  rs   │  rt   │  rd   │ shamt │ funct  │                   │
│    │  =0  │ =$s1  │ =$s2  │ =$t0  │  =0   │ =add   │                   │
│    └──────┴───────┴───────┴───────┴───────┴────────┘                   │
│    = 0x02324020                                                         │
│                                                                         │
│    Example 2: LW $t0, 100($s0)                                          │
│    ────────────────────────────────────────────────                     │
│    I-type: rt = Memory[rs + offset]                                     │
│    $t0 = 8, $s0 = 16, offset = 100                                      │
│                                                                         │
│    ┌──────┬───────┬───────┬─────────────────────┐                      │
│    │100011│ 10000 │ 01000 │ 0000000001100100    │                      │
│    │  op  │  rs   │  rt   │    immediate        │                      │
│    │ =lw  │ =$s0  │ =$t0  │      =100           │                      │
│    └──────┴───────┴───────┴─────────────────────┘                      │
│    = 0x8E080064                                                         │
│                                                                         │
│    Example 3: BEQ $s0, $s1, loop  (loop is 8 bytes ahead)               │
│    ────────────────────────────────────────────────                     │
│    I-type: if (rs == rt) PC = PC + 4 + offset*4                         │
│    offset = (loop - PC - 4) / 4 = 2                                     │
│                                                                         │
│    ┌──────┬───────┬───────┬─────────────────────┐                      │
│    │000100│ 10000 │ 10001 │ 0000000000000010    │                      │
│    │  op  │  rs   │  rt   │    offset           │                      │
│    │ =beq │ =$s0  │ =$s1  │      =2             │                      │
│    └──────┴───────┴───────┴─────────────────────┘                      │
│    = 0x12110002                                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 ARM Instruction Formats

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ARM Instruction Formats (32-bit)                    │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                    Data Processing                              │ │
│    │                                                                 │ │
│    │   31  28 27 26 25 24   21 20 19  16 15  12 11           0      │ │
│    │   ┌────┬─────┬──┬───────┬──┬──────┬──────┬───────────────┐     │ │
│    │   │cond│ 00  │I │opcode │S │  Rn  │  Rd  │   Operand2    │     │ │
│    │   │4bit│2bit │1 │ 4bit  │1 │ 4bit │ 4bit │    12bit      │     │ │
│    │   └────┴─────┴──┴───────┴──┴──────┴──────┴───────────────┘     │ │
│    │                                                                 │ │
│    │   cond: Condition code (EQ, NE, GT, LT, etc.)                   │ │
│    │   I: Immediate flag                                             │ │
│    │   S: Update flags flag                                          │ │
│    │                                                                 │ │
│    │   Example: ADD R0, R1, R2                                       │ │
│    │       Rd = Rn + Operand2                                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                    Load/Store (Memory Access)                   │ │
│    │                                                                 │ │
│    │   31  28 27 26 25 24 23 22 21 20 19  16 15  12 11         0    │ │
│    │   ┌────┬─────┬──┬──┬──┬──┬──┬──┬──────┬──────┬─────────────┐   │ │
│    │   │cond│ 01  │I │P │U │B │W │L │  Rn  │  Rd  │   offset    │   │ │
│    │   │4bit│2bit │1 │1 │1 │1 │1 │1 │ 4bit │ 4bit │   12bit     │   │ │
│    │   └────┴─────┴──┴──┴──┴──┴──┴──┴──────┴──────┴─────────────┘   │ │
│    │                                                                 │ │
│    │   P: Pre/Post indexing                                          │ │
│    │   U: Up/Down (add/subtract)                                     │ │
│    │   B: Byte/Word                                                  │ │
│    │   W: Write-back                                                 │ │
│    │   L: Load/Store                                                 │ │
│    │                                                                 │ │
│    │   Example: LDR R0, [R1, #100]                                   │ │
│    │       R0 = Memory[R1 + 100]                                     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                         Branch                                  │ │
│    │                                                                 │ │
│    │   31  28 27  25 24 23                            0             │ │
│    │   ┌────┬───────┬──┬──────────────────────────────┐             │ │
│    │   │cond│  101  │L │          offset              │             │ │
│    │   │4bit│ 3bit  │1 │          24bit               │             │ │
│    │   └────┴───────┴──┴──────────────────────────────┘             │ │
│    │                                                                 │ │
│    │   L: Link (1 for BL, saves return address to LR)                │ │
│    │   offset: PC-relative offset (<<2 then sign-extended)           │ │
│    │                                                                 │ │
│    │   Example: BL function                                          │ │
│    │       LR = PC + 4; PC = PC + offset << 2                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 x86 Instruction Format

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   x86 Instruction Format (Variable Length)               │
│                                                                         │
│    ┌──────┬───────┬─────────┬────────┬─────────────┬───────────────┐   │
│    │Prefix│Opcode │ ModR/M  │  SIB   │Displacement │  Immediate    │   │
│    │0-4   │1-3    │ 0-1     │ 0-1    │   0,1,2,4   │  0,1,2,4      │   │
│    │bytes │bytes  │ byte    │ byte   │   bytes     │  bytes        │   │
│    └──────┴───────┴─────────┴────────┴─────────────┴───────────────┘   │
│                                                                         │
│    ModR/M byte (operand specification):                                 │
│    ┌────────┬───────┬───────┐                                          │
│    │  Mod   │  Reg  │  R/M  │                                          │
│    │ 2 bits │3 bits │3 bits │                                          │
│    └────────┴───────┴───────┘                                          │
│                                                                         │
│    SIB byte (complex address calculation):                              │
│    ┌────────┬───────┬───────┐                                          │
│    │ Scale  │ Index │ Base  │                                          │
│    │ 2 bits │3 bits │3 bits │                                          │
│    └────────┴───────┴───────┘                                          │
│    Address = Base + Index * Scale + Displacement                        │
│                                                                         │
│    Example:                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │ MOV EAX, [EBX+ECX*4+100]                                        │ │
│    │                                                                 │ │
│    │ ┌────────┬─────────┬─────────┬──────────────┐                   │ │
│    │ │ 8B     │ 84      │ 8B      │ 64 00 00 00  │                   │ │
│    │ │ Opcode │ ModR/M  │ SIB     │ Displacement │                   │ │
│    │ │        │Mod=10   │Scale=4  │ =100         │                   │ │
│    │ │        │Reg=EAX  │Index=ECX│              │                   │ │
│    │ │        │R/M=100  │Base=EBX │              │                   │ │
│    │ └────────┴─────────┴─────────┴──────────────┘                   │ │
│    │ Total: 7 bytes                                                  │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Addressing Modes

### 4.1 Addressing Modes Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Types of Addressing Modes                        │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  1. Immediate Addressing                                        │ │
│    │                                                                 │ │
│    │     Instruction: ADDI $t0, $t1, 100                             │ │
│    │                                                                 │ │
│    │     ┌────────┬───────────────────────────────┐                  │ │
│    │     │ Opcode │  ... │ 100 (Immediate Value) │                  │ │
│    │     └────────┴─────────────────┬─────────────┘                  │ │
│    │                                │                                │ │
│    │                                ▼                                │ │
│    │                            Operand                              │ │
│    │                                                                 │ │
│    │     Operand is included in the instruction                      │ │
│    │     Fast, suitable for constants                                │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  2. Register Addressing                                         │ │
│    │                                                                 │ │
│    │     Instruction: ADD $t0, $t1, $t2                              │ │
│    │                                                                 │ │
│    │     ┌────────┬──────┬──────┬──────┐                             │ │
│    │     │ Opcode │ $t1  │ $t2  │ $t0  │                             │ │
│    │     └────────┴───┬──┴───┬──┴──────┘                             │ │
│    │                  │      │                                       │ │
│    │                  ▼      ▼                                       │ │
│    │            ┌──────────────────┐                                 │ │
│    │            │  Register File   │                                 │ │
│    │            │  ┌────┬────┬───┐│                                 │ │
│    │            │  │$t1 │$t2 │...││                                 │ │
│    │            │  └──┬─┴──┬─┴───┘│                                 │ │
│    │            └─────┼────┼──────┘                                 │ │
│    │                  ▼    ▼                                        │ │
│    │               Operands                                          │ │
│    │                                                                 │ │
│    │     Register number specifies operand                           │ │
│    │     Fastest (register access)                                   │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Memory Addressing Modes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Memory Addressing Modes                           │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  3. Direct Addressing                                           │ │
│    │                                                                 │ │
│    │     Instruction: LOAD R1, 0x1000                                │ │
│    │                                                                 │ │
│    │     ┌────────┬──────────────────────┐                           │ │
│    │     │ Opcode │ Address: 0x1000      │                           │ │
│    │     └────────┴──────────┬───────────┘                           │ │
│    │                         │                                       │ │
│    │                         ▼                                       │ │
│    │               ┌─────────────────┐                               │ │
│    │               │     Memory      │                               │ │
│    │               │  0x1000: [data] │ ───► Operand                  │ │
│    │               └─────────────────┘                               │ │
│    │                                                                 │ │
│    │     Address directly included in instruction                    │ │
│    │     Used for global variables                                   │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  4. Indirect Addressing                                         │ │
│    │                                                                 │ │
│    │     Instruction: LOAD R1, (0x1000)                              │ │
│    │                                                                 │ │
│    │     ┌────────┬──────────────────────┐                           │ │
│    │     │ Opcode │ Address: 0x1000      │                           │ │
│    │     └────────┴──────────┬───────────┘                           │ │
│    │                         │                                       │ │
│    │                         ▼                                       │ │
│    │               ┌─────────────────┐                               │ │
│    │               │     Memory      │                               │ │
│    │               │  0x1000: 0x2000 │ ─┐                            │ │
│    │               │  0x2000: [data] │◄┘ ───► Operand               │ │
│    │               └─────────────────┘                               │ │
│    │                                                                 │ │
│    │     Actual address read from memory                             │ │
│    │     Used for pointers, dynamic data structures                  │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  5. Register Indirect Addressing                                │ │
│    │                                                                 │ │
│    │     Instruction: LW $t0, ($s0)   ; MIPS                         │ │
│    │     Instruction: MOV EAX, [EBX]  ; x86                          │ │
│    │                                                                 │ │
│    │     ┌────────┬───────┐                                          │ │
│    │     │ Opcode │  $s0  │                                          │ │
│    │     └────────┴───┬───┘                                          │ │
│    │                  │                                              │ │
│    │                  ▼                                              │ │
│    │         ┌──────────────┐                                        │ │
│    │         │ $s0 = 0x1000 │                                        │ │
│    │         └──────┬───────┘                                        │ │
│    │                │                                                │ │
│    │                ▼                                                │ │
│    │         ┌─────────────────┐                                     │ │
│    │         │     Memory      │                                     │ │
│    │         │ 0x1000: [data]  │ ───► Operand                        │ │
│    │         └─────────────────┘                                     │ │
│    │                                                                 │ │
│    │     Register contains the memory address                        │ │
│    │     Used for pointer dereferencing                              │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Displacement Addressing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Displacement Addressing Modes                      │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  6. Displacement/Base Addressing                                │ │
│    │                                                                 │ │
│    │     Instruction: LW $t0, 100($s0)   ; MIPS                      │ │
│    │     Instruction: MOV EAX, [EBX+100] ; x86                       │ │
│    │                                                                 │ │
│    │     ┌────────┬───────┬───────────┐                              │ │
│    │     │ Opcode │  $s0  │ offset=100│                              │ │
│    │     └────────┴───┬───┴─────┬─────┘                              │ │
│    │                  │         │                                    │ │
│    │                  ▼         │                                    │ │
│    │         ┌──────────────┐   │                                    │ │
│    │         │ $s0 = 0x1000 │   │                                    │ │
│    │         └──────┬───────┘   │                                    │ │
│    │                │           │                                    │ │
│    │                └─────┬─────┘                                    │ │
│    │                      │ + (address calculation)                  │ │
│    │                      ▼                                          │ │
│    │              Effective Address = 0x1064                         │ │
│    │                      │                                          │ │
│    │                      ▼                                          │ │
│    │         ┌─────────────────┐                                     │ │
│    │         │     Memory      │                                     │ │
│    │         │ 0x1064: [data]  │ ───► Operand                        │ │
│    │         └─────────────────┘                                     │ │
│    │                                                                 │ │
│    │     Base register + displacement calculates address             │ │
│    │     Used for array, struct access                               │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  7. Indexed Addressing                                          │ │
│    │                                                                 │ │
│    │     Instruction: MOV EAX, [EBX + ECX*4 + 100] ; x86             │ │
│    │                                                                 │ │
│    │     Effective Address = Base + Index × Scale + Displacement     │ │
│    │                       = EBX + ECX × 4 + 100                     │ │
│    │                                                                 │ │
│    │     ┌────────────────────────────────────────┐                  │ │
│    │     │                                        │                  │ │
│    │     │   EBX (base) ──────────────┐           │                  │ │
│    │     │                            │           │                  │ │
│    │     │   ECX (index) ─► × 4 ──────┤           │                  │ │
│    │     │                            │ + ──► Effective Address     │ │
│    │     │   100 (displacement) ──────┘           │                  │ │
│    │     │                                        │                  │ │
│    │     └────────────────────────────────────────┘                  │ │
│    │                                                                 │ │
│    │     Optimized for array element access                          │ │
│    │     Example: array[i] = arr_base + i * sizeof(element)          │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 PC-Relative Addressing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PC-Relative Addressing                           │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  8. PC-Relative Addressing                                      │ │
│    │                                                                 │ │
│    │     Instruction: BEQ $t0, $t1, label                            │ │
│    │                                                                 │ │
│    │     Effective Address = PC + offset × 4                         │ │
│    │                                                                 │ │
│    │     ┌────────┬───────┬───────┬───────────────┐                  │ │
│    │     │ Opcode │  $t0  │  $t1  │ offset = 3    │                  │ │
│    │     └────────┴───────┴───────┴───────┬───────┘                  │ │
│    │                                      │                          │ │
│    │     Memory layout:                   │                          │ │
│    │     ┌─────────────────────────────┐  │                          │ │
│    │     │ 0x1000: BEQ ...        ◄─── PC (current)                  │ │
│    │     │ 0x1004: ...                │                              │ │
│    │     │ 0x1008: ...                │                              │ │
│    │     │ 0x100C: ...                │                              │ │
│    │     │ 0x1010: label:        ◄─── PC + 4 + 3×4 = 0x1010          │ │
│    │     └─────────────────────────────┘                             │ │
│    │                                                                 │ │
│    │     Primarily used for branch instructions                      │ │
│    │     Supports Position Independent Code (PIC)                    │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Addressing Modes Comparison

| Mode | Effective Address | Advantages | Disadvantages | Use Case |
|------|-------------------|------------|---------------|----------|
| Immediate | None (value itself) | Fast | Limited value size | Constants |
| Register | Register | Fastest | Limited registers | Temp variables |
| Direct | Address in instruction | Simple | Address size limit | Global variables |
| Indirect | Mem[address] | Flexible | 2 memory accesses | Pointers |
| Register Indirect | Reg | Flexible | - | Pointer dereference |
| Displacement | Reg + offset | Array access | - | Struct/Array |
| Indexed | Base + Idx×S + D | Array optimization | Complex | array[i] |
| PC-Relative | PC + offset | PIC support | Range limited | Branches |

---

## 5. Major ISAs

### 5.1 x86 / x86-64

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           x86 / x86-64                                   │
│                                                                         │
│    History:                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  1978: 8086 (16-bit)                                            │ │
│    │  1985: 80386 (32-bit, IA-32)                                    │ │
│    │  2003: x86-64 / AMD64 (64-bit)                                  │ │
│    │  Today: Used in billions of PCs/servers                         │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Registers (x86-64):                                                  │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  General-purpose registers (16):                                │ │
│    │  RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP, R8-R15                │ │
│    │                                                                 │ │
│    │  Special registers:                                             │ │
│    │  RIP (Instruction Pointer), RFLAGS (status flags)              │ │
│    │                                                                 │ │
│    │  Segment registers:                                             │ │
│    │  CS, DS, SS, ES, FS, GS                                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Representative instructions:                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; Data movement                                                │ │
│    │  MOV EAX, EBX          ; EAX = EBX                             │ │
│    │  MOV EAX, [EBX]        ; EAX = Memory[EBX]                     │ │
│    │  MOV EAX, [EBX+ECX*4]  ; Array access                          │ │
│    │                                                                 │ │
│    │  ; Arithmetic                                                   │ │
│    │  ADD EAX, EBX          ; EAX = EAX + EBX                       │ │
│    │  SUB EAX, 10           ; EAX = EAX - 10                        │ │
│    │  IMUL EAX, EBX         ; EAX = EAX * EBX                       │ │
│    │                                                                 │ │
│    │  ; Branching                                                    │ │
│    │  CMP EAX, EBX          ; Compare (set flags)                   │ │
│    │  JE  label             ; Jump if Equal                         │ │
│    │  JNE label             ; Jump if Not Equal                     │ │
│    │  JMP label             ; Unconditional jump                    │ │
│    │                                                                 │ │
│    │  ; Function calls                                               │ │
│    │  CALL function         ; Call function                         │ │
│    │  RET                   ; Return                                │ │
│    │  PUSH EAX              ; Push to stack                         │ │
│    │  POP  EBX              ; Pop from stack                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 ARM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                               ARM                                        │
│                                                                         │
│    History:                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  1985: ARM1 (32-bit RISC)                                       │ │
│    │  2011: ARMv8 (64-bit, AArch64)                                  │ │
│    │  Today: Smartphones, tablets, IoT, servers (Apple M1→M4,        │ │
│    │         AWS Graviton4)                                          │ │
│    │  Characteristic: Low power, license-based business              │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Registers (AArch64):                                                 │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  General-purpose registers (31):                                │ │
│    │  X0-X30 (64-bit), W0-W30 (lower 32 bits)                       │ │
│    │                                                                 │ │
│    │  Special registers:                                             │ │
│    │  SP (Stack Pointer), PC (Program Counter)                      │ │
│    │  X30/LR (Link Register)                                        │ │
│    │  XZR/WZR (Zero Register)                                       │ │
│    │                                                                 │ │
│    │  System registers:                                              │ │
│    │  NZCV (condition flags), FPCR, FPSR                            │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Representative instructions (AArch64):                               │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; Data movement                                                │ │
│    │  MOV  X0, X1           ; X0 = X1                               │ │
│    │  LDR  X0, [X1]         ; X0 = Memory[X1]                       │ │
│    │  STR  X0, [X1, #8]     ; Memory[X1+8] = X0                     │ │
│    │                                                                 │ │
│    │  ; Arithmetic                                                   │ │
│    │  ADD  X0, X1, X2       ; X0 = X1 + X2                          │ │
│    │  SUB  X0, X1, #10      ; X0 = X1 - 10                          │ │
│    │  MUL  X0, X1, X2       ; X0 = X1 * X2                          │ │
│    │                                                                 │ │
│    │  ; Compare and branch                                           │ │
│    │  CMP  X0, X1           ; X0 - X1 (set flags)                   │ │
│    │  B.EQ label            ; Branch if Equal                       │ │
│    │  B.NE label            ; Branch if Not Equal                   │ │
│    │  B    label            ; Unconditional branch                  │ │
│    │                                                                 │ │
│    │  ; Function calls                                               │ │
│    │  BL   function         ; Branch with Link (return addr to LR)  │ │
│    │  RET                   ; Return (jump to LR)                   │ │
│    │                                                                 │ │
│    │  ; Conditional execution (ARM32)                                │ │
│    │  ADDEQ R0, R1, R2      ; ADD only if Equal                     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Features:                                                            │
│    - Conditional execution (condition suffix on most instructions)      │
│    - Load/Store architecture                                            │
│    - Thumb instruction set (16-bit, improved code density)              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 MIPS

```
┌─────────────────────────────────────────────────────────────────────────┐
│                               MIPS                                       │
│                                                                         │
│    History and Features:                                                │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  1985: MIPS R2000 (pure RISC design)                            │ │
│    │  Feature: Academically important (standard for CS education)    │ │
│    │  Uses: Embedded systems, network equipment, game consoles       │ │
│    │        (PS1/2)                                                  │ │
│    │  2021: MIPS Technologies transitioned to RISC-V                 │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Registers (32):                                                      │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  $zero (R0)  : Always 0                                        │ │
│    │  $at   (R1)  : Assembler temporary                             │ │
│    │  $v0-v1 (R2-3): Function return values                         │ │
│    │  $a0-a3 (R4-7): Function arguments                             │ │
│    │  $t0-t7 (R8-15): Temporaries (caller-saved)                    │ │
│    │  $s0-s7 (R16-23): Saved (callee-saved)                         │ │
│    │  $t8-t9 (R24-25): Additional temporaries                       │ │
│    │  $gp   (R28) : Global pointer                                  │ │
│    │  $sp   (R29) : Stack pointer                                   │ │
│    │  $fp   (R30) : Frame pointer                                   │ │
│    │  $ra   (R31) : Return address                                  │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Representative instructions:                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; R-type (register)                                            │ │
│    │  add  $t0, $t1, $t2    ; $t0 = $t1 + $t2                       │ │
│    │  sub  $t0, $t1, $t2    ; $t0 = $t1 - $t2                       │ │
│    │  and  $t0, $t1, $t2    ; $t0 = $t1 & $t2                       │ │
│    │  sll  $t0, $t1, 2      ; $t0 = $t1 << 2                        │ │
│    │                                                                 │ │
│    │  ; I-type (immediate)                                           │ │
│    │  addi $t0, $t1, 100    ; $t0 = $t1 + 100                       │ │
│    │  lw   $t0, 4($sp)      ; $t0 = Memory[$sp + 4]                 │ │
│    │  sw   $t0, 0($sp)      ; Memory[$sp] = $t0                     │ │
│    │  beq  $t0, $t1, label  ; if ($t0 == $t1) goto label            │ │
│    │  bne  $t0, $t1, label  ; if ($t0 != $t1) goto label            │ │
│    │                                                                 │ │
│    │  ; J-type (jump)                                                │ │
│    │  j    target           ; goto target                           │ │
│    │  jal  function         ; $ra = PC+4; goto function             │ │
│    │  jr   $ra              ; goto $ra (function return)            │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 RISC-V

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              RISC-V                                      │
│                                                                         │
│    History and Features:                                                │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  2010: Development started at UC Berkeley                       │ │
│    │  Features:                                                      │ │
│    │  - Open-source ISA (royalty-free)                               │ │
│    │  - Modular design (base + extensions)                           │ │
│    │  - Rapidly adopted by academia and industry                     │ │
│    │  - 32/64/128-bit support                                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Modular Structure:                                                   │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  RV32I / RV64I: Base integer instructions (required)            │ │
│    │  M: Multiplication/Division extension                           │ │
│    │  A: Atomic operations extension                                 │ │
│    │  F: Single-precision floating-point                             │ │
│    │  D: Double-precision floating-point                             │ │
│    │  C: Compressed instructions (16-bit)                            │ │
│    │                                                                 │ │
│    │  Example: RV64IMAFDC = 64-bit + all standard extensions         │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Registers (32):                                                      │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  x0 (zero): Always 0                                           │ │
│    │  x1 (ra)  : Return address                                     │ │
│    │  x2 (sp)  : Stack pointer                                      │ │
│    │  x3 (gp)  : Global pointer                                     │ │
│    │  x4 (tp)  : Thread pointer                                     │ │
│    │  x5-x7    : Temporaries                                        │ │
│    │  x8 (s0/fp): Saved/Frame pointer                               │ │
│    │  x9       : Saved                                              │ │
│    │  x10-x17  : Function arguments/return                          │ │
│    │  x18-x27  : Saved                                              │ │
│    │  x28-x31  : Temporaries                                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Representative instructions:                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; R-type                                                       │ │
│    │  add  x1, x2, x3       ; x1 = x2 + x3                          │ │
│    │  sub  x1, x2, x3       ; x1 = x2 - x3                          │ │
│    │                                                                 │ │
│    │  ; I-type                                                       │ │
│    │  addi x1, x2, 100      ; x1 = x2 + 100                         │ │
│    │  lw   x1, 0(x2)        ; x1 = Memory[x2]                       │ │
│    │                                                                 │ │
│    │  ; S-type (Store)                                               │ │
│    │  sw   x1, 0(x2)        ; Memory[x2] = x1                       │ │
│    │                                                                 │ │
│    │  ; B-type (Branch)                                              │ │
│    │  beq  x1, x2, label    ; if (x1 == x2) goto label              │ │
│    │  bne  x1, x2, label    ; if (x1 != x2) goto label              │ │
│    │                                                                 │ │
│    │  ; J-type                                                       │ │
│    │  jal  x1, target       ; x1 = PC+4; goto target                │ │
│    │  jalr x1, x2, 0        ; x1 = PC+4; goto x2                    │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.5 ISA Comparison Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ISA Comparison Summary                             │
├──────────────┬──────────────┬──────────────┬──────────────┬────────────────┤
│   Feature    │    x86-64    │     ARM      │     MIPS     │    RISC-V      │
├──────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│ Type         │ CISC         │ RISC         │ RISC         │ RISC           │
├──────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│ Instr. length│ 1-15 bytes   │ 4 bytes      │ 4 bytes      │ 4 bytes        │
│              │ (variable)   │ (fixed)      │ (fixed)      │ (2/4 bytes)    │
├──────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│ GP Registers │ 16           │ 31           │ 32           │ 32             │
├──────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│ Endianness   │ Little       │ Bi-endian    │ Bi-endian    │ Little         │
├──────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│ Primary use  │ PC, Server   │ Mobile, IoT  │ Embedded     │ General,       │
│              │              │ Server (now) │ (legacy)     │ Education, IoT │
├──────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│ License      │ Intel/AMD    │ ARM Holdings │ MIPS Tech.   │ Open-source    │
│              │ proprietary  │ license      │ license      │ (free)         │
└──────────────┴──────────────┴──────────────┴──────────────┴────────────────┘
```

---

## 6. Practice Problems

### Basic Problems

1. List 5 things that an ISA defines.

2. Explain 3 major differences between CISC and RISC.

3. Explain the three MIPS instruction formats (R, I, J).

### Instruction Encoding Problems

4. Encode the following MIPS instruction as a 32-bit binary number:
   ```
   ADD $t2, $s0, $s1
   ```
   (Reference: $t2=10, $s0=16, $s1=17, ADD funct=0x20)

5. Encode the following MIPS instruction as a 32-bit binary number:
   ```
   LW $t0, 200($s2)
   ```
   (Reference: $t0=8, $s2=18, LW opcode=0x23)

### Addressing Mode Problems

6. Identify the addressing mode for each instruction:
   - (a) `ADDI $t0, $t1, 100`
   - (b) `LW $t0, 0($s0)`
   - (c) `ADD $t0, $t1, $t2`
   - (d) `J 0x00400000`
   - (e) `BEQ $t0, $t1, label`

7. Explain how the effective address is calculated for x86 `MOV EAX, [EBX + ECX*4 + 100]`.

### Advanced Problems

8. Explain why modern x86 processors internally use RISC-style micro-ops.

9. Describe 3 advantages of RISC-V as an open-source ISA.

10. Convert the following C code to MIPS assembly:
    ```c
    int a = 10;
    int b = 20;
    int c = a + b;
    ```

<details>
<summary>Answers</summary>

1. What an ISA defines:
   - Instructions
   - Registers
   - Data Types
   - Addressing Modes
   - Memory Model

2. CISC vs RISC:
   - Instruction complexity: CISC is complex, RISC is simple
   - Instruction length: CISC is variable, RISC is fixed
   - Memory access: CISC allows all instructions, RISC uses Load/Store only

3. MIPS instruction formats:
   - R-type: Register-to-register operations (ADD, SUB, etc.)
   - I-type: Immediate value operations (ADDI, LW, SW, BEQ, etc.)
   - J-type: Jump instructions (J, JAL)

4. ADD $t2, $s0, $s1 encoding:
   ```
   opcode(6) | rs(5)  | rt(5)  | rd(5)  | shamt(5) | funct(6)
   000000    | 10000  | 10001  | 01010  | 00000    | 100000
   = 0x02115020
   ```

5. LW $t0, 200($s2) encoding:
   ```
   opcode(6) | rs(5)  | rt(5)  | immediate(16)
   100011    | 10010  | 01000  | 0000000011001000
   = 0x8E4800C8
   ```

6. Addressing modes:
   - (a) Immediate Addressing
   - (b) Base/Displacement Addressing
   - (c) Register Addressing
   - (d) Direct Addressing
   - (e) PC-Relative Addressing

7. x86 effective address calculation:
   Effective Address = Base(EBX) + Index(ECX) × Scale(4) + Displacement(100)

8. Reasons for using micro-ops:
   - Pipeline optimization (simple RISC-style instructions)
   - Easier superscalar execution
   - Easier out-of-order execution implementation
   - Maintain backward compatibility while optimizing internally

9. RISC-V open-source advantages:
   - Royalty-free (cost savings)
   - Customizable (optimized for specific purposes)
   - Useful for academic research and education
   - No vendor lock-in

10. C code to MIPS:
    ```mips
    # a = 10
    li   $s0, 10        # $s0 = 10
    # b = 20
    li   $s1, 20        # $s1 = 20
    # c = a + b
    add  $s2, $s0, $s1  # $s2 = $s0 + $s1
    ```

</details>

---

## Next Steps

- [10_Assembly_Language_Basics.md](./10_Assembly_Language_Basics.md) - x86/ARM basics, fundamental instructions

---

## References

- Computer Organization and Design: MIPS Edition (Patterson & Hennessy)
- Computer Organization and Design: RISC-V Edition (Patterson & Hennessy)
- [MIPS Instruction Reference](https://www.mips.com/products/architectures/)
- [ARM Architecture Reference Manual](https://developer.arm.com/documentation)
- [RISC-V Specifications](https://riscv.org/technical/specifications/)
- [Intel x86 Developer Manuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
