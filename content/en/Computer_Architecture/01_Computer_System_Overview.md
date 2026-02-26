# Computer System Overview

**Previous**: [Overview](./00_Overview.md) | **Next**: [Data Representation Basics](./02_Data_Representation_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the generational evolution of computers from vacuum tubes to modern VLSI
2. Explain von Neumann architecture and its core principles (stored program, sequential execution, binary representation)
3. Identify the von Neumann bottleneck and describe methods to mitigate it
4. Identify the major hardware components (CPU, memory, I/O) and their roles
5. Distinguish between system software and application software
6. Calculate basic performance metrics (CPI, MIPS, execution time) using standard formulas
7. Apply Amdahl's Law to evaluate the impact of partial system improvements

---

Every device you use -- from your phone to cloud servers -- is built on the same fundamental architecture that John von Neumann described in 1945. Understanding how a computer's hardware components work together is the foundation for writing efficient software, diagnosing performance problems, and appreciating why certain design decisions were made.

## Table of Contents

1. [History of Computers](#1-history-of-computers)
2. [Von Neumann Architecture](#2-von-neumann-architecture)
3. [Hardware Components](#3-hardware-components)
4. [Software Layers](#4-software-layers)
5. [Performance Measurement](#5-performance-measurement)
6. [Practice Problems](#6-practice-problems)

---

## 1. History of Computers

### Evolution by Generation

| Generation | Period | Core Technology | Characteristics |
|-----------|--------|----------------|----------------|
| 1st Gen | 1940s-1950s | Vacuum tubes | ENIAC, large size, high heat, low reliability |
| 2nd Gen | 1950s-1960s | Transistors | Miniaturization, low power, COBOL/FORTRAN |
| 3rd Gen | 1960s-1970s | Integrated Circuits (IC) | Operating systems, multiprogramming |
| 4th Gen | 1970s-Present | VLSI/ULSI | Microprocessors, personal computers |
| 5th Gen | Present-Future | AI/Quantum computers | Parallel processing, artificial intelligence |

### Major Milestones

```
1946: ENIAC (first general-purpose electronic computer)
1947: Transistor invented (Bell Labs)
1958: Integrated circuit invented (Jack Kilby)
1971: Intel 4004 (first commercial microprocessor)
1981: IBM PC
2007: iPhone (mobile computing era)
```

---

## 2. Von Neumann Architecture

### Core Concept

The computer architecture proposed by John von Neumann in 1945, which most modern computers follow.

```
┌─────────────────────────────────────────┐
│              CPU (Central Processing Unit) │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Control │  │   ALU   │  │Registers│  │
│  │  Unit   │  │(Arithmetic│  │         │  │
│  │         │  │  Logic  │  │         │  │
│  │         │  │  Unit)  │  │         │  │
│  └─────────┘  └─────────┘  └─────────┘  │
└──────────────────┬──────────────────────┘
                   │ System Bus
     ┌─────────────┼─────────────┐
     │             │             │
┌────┴────┐  ┌─────┴─────┐  ┌────┴────┐
│ Memory  │  │   Input   │  │  Output │
│  (RAM)  │  │ Devices   │  │ Devices │
│         │  │(Keyboard, │  │(Monitor,│
│         │  │   etc.)   │  │  etc.)  │
└─────────┘  └───────────┘  └─────────┘
```

### Core Principles

1. **Stored Program Concept**: Programs and data stored in the same memory
2. **Sequential Execution**: Instructions executed sequentially, one at a time
3. **Binary Representation**: All data represented in binary

### Von Neumann Bottleneck

```
Data transfer speed between CPU and memory is a performance bottleneck

CPU Speed >> Memory Speed

Solutions:
- Cache memory
- Pipelining
- Multiple buses
```

---

## 3. Hardware Components

### 3.1 Central Processing Unit (CPU)

```
┌────────────────────────────────────────┐
│                  CPU                    │
│  ┌──────────────────────────────────┐  │
│  │      Control Unit (CU)            │  │
│  │  - Instruction decoding           │  │
│  │  - Execution sequence control     │  │
│  │  - Control signal generation      │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │  Arithmetic Logic Unit (ALU)      │  │
│  │  - Arithmetic operations (+,-,*,/)│  │
│  │  - Logic operations (AND,OR,NOT)  │  │
│  │  - Comparison operations          │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │         Registers                 │  │
│  │  - PC (Program Counter)           │  │
│  │  - IR (Instruction Register)      │  │
│  │  - AC (Accumulator)               │  │
│  │  - MAR, MBR (Memory access)       │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

### 3.2 Main Memory

| Type | Characteristics | Usage |
|------|----------------|-------|
| RAM | Volatile, read/write | Running programs/data |
| ROM | Non-volatile, read-only | BIOS, firmware |
| Cache | High-speed, small capacity | Bridge CPU-memory speed gap |

### 3.3 Secondary Storage

| Type | Speed | Capacity | Characteristics |
|------|-------|----------|----------------|
| SSD | Fast | Medium | Flash memory, silent |
| HDD | Slow | Large | Magnetic disk, inexpensive |
| USB | Medium | Small | Portable |

### 3.4 Bus System

```
┌───────────────────────────────────────────┐
│              Data Bus                      │
│    (Data transfer between CPU ↔ Memory)   │
├───────────────────────────────────────────┤
│              Address Bus                   │
│    (Memory address specification, unidirectional) │
├───────────────────────────────────────────┤
│              Control Bus                   │
│    (Read/write control signals)           │
└───────────────────────────────────────────┘
```

---

## 4. Software Layers

### Layer Structure

```
┌─────────────────────────────────┐
│    Application Programs          │  Web browsers, games, word processors
├─────────────────────────────────┤
│    System Software               │  Compilers, libraries
├─────────────────────────────────┤
│    Operating System (OS)         │  Windows, Linux, macOS
├─────────────────────────────────┤
│    Firmware / BIOS               │  Hardware initialization
├─────────────────────────────────┤
│    Hardware                      │  CPU, memory, disk
└─────────────────────────────────┘
```

### Instruction Execution Cycle

```
┌───────────────────────────────────────────────┐
│                                               │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │  Fetch  │ → │ Decode  │ → │ Execute │     │
│  │         │   │         │   │         │     │
│  └─────────┘   └─────────┘   └─────────┘     │
│       ↑                            │          │
│       └────────────────────────────┘          │
│                  Repeat                        │
└───────────────────────────────────────────────┘

1. Fetch:   Retrieve instruction from address pointed to by PC
2. Decode:  Decode instruction, determine required operation
3. Execute: Perform operation in ALU, store result
```

---

## 5. Performance Measurement

### Key Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| Clock Speed | Clock cycles per second | Hz (GHz) |
| CPI | Cycles per instruction | cycles/instruction |
| MIPS | Million instructions per second | million instructions/sec |
| FLOPS | Floating-point operations per second | floating point ops/sec |

### Performance Calculation Formulas

```
CPU Time = Instruction Count × CPI × Clock Period

         Instruction Count × CPI
CPU Time = ───────────────────────
            Clock Speed

Example:
- Instruction count: 1 billion
- CPI: 2
- Clock speed: 4GHz

CPU Time = (10^9 × 2) / (4 × 10^9) = 0.5 seconds
```

### Amdahl's Law

```
Overall Speedup = 1 / ((1 - P) + P/S)

P: Fraction of program that can be improved
S: Speedup factor for that fraction

Example: Improve 80% of program to run 2× faster
= 1 / ((1 - 0.8) + 0.8/2)
= 1 / (0.2 + 0.4)
= 1.67× speedup
```

---

## 6. Practice Problems

### Basic Problems

1. What are the five components of Von Neumann architecture?

2. Which of the following is volatile memory?
   - (a) ROM
   - (b) RAM
   - (c) SSD
   - (d) HDD

3. Which part of the CPU decodes instructions?

### Calculation Problems

4. If a CPU has a clock speed of 3GHz and CPI of 1.5, how long does it take to execute 900 million instructions?

5. If 70% of a program can be parallelized and runs on 4 cores, what is the maximum speedup according to Amdahl's Law?

### Advanced Problems

6. Describe three methods to solve the Von Neumann bottleneck.

7. Explain the differences between Harvard architecture and Von Neumann architecture.

<details>
<summary>Answers</summary>

1. Input devices, output devices, memory, arithmetic logic unit (ALU), control unit

2. (b) RAM

3. Control Unit

4. (9×10^8 × 1.5) / (3×10^9) = 0.45 seconds

5. 1 / ((1-0.7) + 0.7/4) = 1 / (0.3 + 0.175) = 2.1×

6. Cache memory, pipelining, multiple buses, Harvard architecture

7. Harvard architecture stores instructions and data in separate memories and accesses them via separate buses

</details>

---

## Next Steps

- [02_Data_Representation_Basics.md](./02_Data_Representation_Basics.md) - Binary numbers and base conversion

---

## References

- Computer Organization and Design (Patterson & Hennessy)
- [Crash Course: Computer Science](https://www.youtube.com/playlist?list=PL8dPuuaLjXtNlUrzyH5r6jN9ulIgZBpdo)
- [Nand2Tetris](https://www.nand2tetris.org/)
