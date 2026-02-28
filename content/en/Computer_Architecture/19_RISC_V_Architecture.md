# RISC-V Architecture

**Previous**: [Parallel Processing and Multicore](./18_Parallel_Processing_Multicore.md) | **Next**: [Power Management](./20_Power_Management.md)

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: Instruction Set Architecture (Lesson 9), Assembly Language Basics (Lesson 10)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the design philosophy behind RISC-V and why it uses an open ISA model
2. Describe the RV32I base integer instruction set and its register file
3. Differentiate between the standard extensions (M, A, F, D, C, V) and their purposes
4. Compare RISC-V addressing modes, instruction encoding, and calling conventions with x86 and ARM
5. Analyze RISC-V assembly code and trace instruction execution through a pipeline

---

Most instruction set architectures are proprietary: x86 belongs to Intel/AMD, ARM requires licensing from Arm Holdings. RISC-V is different -- it is a free, open ISA designed at UC Berkeley in 2010, and it has since grown from an academic project into a serious contender for embedded systems, data centers, and even supercomputers. Understanding RISC-V reveals what a modern ISA looks like when designed from scratch, without decades of backwards-compatibility baggage.

---

## Table of Contents

1. [History and Design Philosophy](#1-history-and-design-philosophy)
2. [ISA Overview](#2-isa-overview)
3. [RV32I Base Integer Instruction Set](#3-rv32i-base-integer-instruction-set)
4. [Instruction Encoding Formats](#4-instruction-encoding-formats)
5. [Standard Extensions](#5-standard-extensions)
6. [Privilege Levels and CSRs](#6-privilege-levels-and-csrs)
7. [RISC-V vs x86 vs ARM](#7-risc-v-vs-x86-vs-arm)
8. [RISC-V Ecosystem](#8-risc-v-ecosystem)
9. [Practice Problems](#9-practice-problems)

---

## 1. History and Design Philosophy

### 1.1 Origins

RISC-V (pronounced "risk-five") was created in 2010 by Krste Asanović, David Patterson (co-author of the famous Patterson & Hennessy textbooks), and students at UC Berkeley. The "V" stands for the fifth generation of RISC architectures designed at Berkeley (after RISC-I, RISC-II, SOAR, SPUR).

### 1.2 Design Goals

| Goal | How RISC-V Achieves It |
|------|----------------------|
| **Open & Free** | ISA specification is open-source; anyone can implement without licensing fees |
| **Modular** | Small base ISA + optional extensions; no mandatory bloat |
| **Clean** | No legacy instructions from 40 years of backward compatibility |
| **Scalable** | Same ISA for microcontrollers (RV32E) to supercomputers (RV64GC + V) |
| **Stable** | Base ISA is frozen; extensions go through a ratification process |

### 1.3 RISC-V Foundation → RISC-V International

The RISC-V Foundation (now RISC-V International, based in Switzerland for geopolitical neutrality) manages the specification. Over 3,000 member organizations contribute, including Google, NVIDIA, Qualcomm, Samsung, and Western Digital.

---

## 2. ISA Overview

### 2.1 Base ISAs

RISC-V defines multiple base integer ISAs:

| Base ISA | Word Size | Registers | Use Case |
|----------|-----------|-----------|----------|
| **RV32I** | 32-bit | 32 × 32-bit | Embedded, microcontrollers |
| **RV32E** | 32-bit | 16 × 32-bit | Ultra-low-power embedded |
| **RV64I** | 64-bit | 32 × 64-bit | Application processors, servers |
| **RV128I** | 128-bit | 32 × 128-bit | Future (draft specification) |

### 2.2 Register File

RV32I/RV64I has 32 general-purpose integer registers:

```
┌──────────┬──────────┬────────────────────────┐
│ Register │ ABI Name │ Description            │
├──────────┼──────────┼────────────────────────┤
│ x0       │ zero     │ Hardwired to 0         │
│ x1       │ ra       │ Return address         │
│ x2       │ sp       │ Stack pointer          │
│ x3       │ gp       │ Global pointer         │
│ x4       │ tp       │ Thread pointer         │
│ x5-x7    │ t0-t2    │ Temporaries            │
│ x8       │ s0/fp    │ Saved register / Frame │
│ x9       │ s1       │ Saved register         │
│ x10-x11  │ a0-a1    │ Arguments / Return val │
│ x12-x17  │ a2-a7    │ Arguments              │
│ x18-x27  │ s2-s11   │ Saved registers        │
│ x28-x31  │ t3-t6    │ Temporaries            │
└──────────┴──────────┴────────────────────────┘
```

Key design choice: `x0` is hardwired to zero. This eliminates the need for a separate "load immediate zero" instruction -- `add x5, x0, x0` clears x5.

### 2.3 Program Counter

The PC is not part of the general register file (unlike ARM). It is only modified implicitly by branch/jump instructions.

---

## 3. RV32I Base Integer Instruction Set

RV32I has only **47 instructions** -- remarkably small compared to x86 (thousands) or even ARM (hundreds).

### 3.1 Instruction Categories

| Category | Instructions | Examples |
|----------|-------------|---------|
| **Arithmetic** | 10 | `ADD`, `SUB`, `ADDI`, `SLT`, `SLTI` |
| **Logical** | 6 | `AND`, `OR`, `XOR`, `ANDI`, `ORI`, `XORI` |
| **Shift** | 6 | `SLL`, `SRL`, `SRA`, `SLLI`, `SRLI`, `SRAI` |
| **Load** | 5 | `LB`, `LH`, `LW`, `LBU`, `LHU` |
| **Store** | 3 | `SB`, `SH`, `SW` |
| **Branch** | 6 | `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU` |
| **Jump** | 2 | `JAL`, `JALR` |
| **Upper Imm** | 2 | `LUI`, `AUIPC` |
| **System** | 7 | `ECALL`, `EBREAK`, `FENCE`, CSR instructions |

### 3.2 Assembly Examples

```asm
# --- Simple arithmetic ---
addi  x5, x0, 42       # x5 = 0 + 42 = 42
addi  x6, x0, 17       # x6 = 17
add   x7, x5, x6       # x7 = 42 + 17 = 59
sub   x8, x5, x6       # x8 = 42 - 17 = 25

# --- Memory access ---
sw    x7, 0(x2)        # Store x7 to address [sp + 0]
lw    x9, 0(x2)        # Load word from [sp + 0] into x9

# --- Conditional branch ---
beq   x5, x6, equal    # If x5 == x6, jump to 'equal'
blt   x5, x6, less     # If x5 < x6 (signed), jump to 'less'

# --- Function call ---
jal   ra, my_function  # Jump to my_function, save return addr in ra
# ... my_function returns with:
jalr  x0, ra, 0        # Jump to address in ra (return)
```

### 3.3 No Multiply in Base ISA

Unlike x86/ARM, the base RV32I has no multiply or divide instructions. These are in the **M extension**. This keeps the base ISA minimal for the simplest microcontrollers that may not need hardware multiplication.

### 3.4 Pseudo-Instructions

The assembler recognizes common patterns:

| Pseudo-Instruction | Actual Instruction | Meaning |
|--------------------|--------------------|---------|
| `li x5, 100` | `addi x5, x0, 100` | Load immediate |
| `mv x5, x6` | `addi x5, x6, 0` | Register copy |
| `nop` | `addi x0, x0, 0` | No operation |
| `ret` | `jalr x0, ra, 0` | Return from function |
| `j label` | `jal x0, label` | Unconditional jump |
| `beqz x5, label` | `beq x5, x0, label` | Branch if zero |
| `not x5, x6` | `xori x5, x6, -1` | Bitwise NOT |
| `neg x5, x6` | `sub x5, x0, x6` | Negate |

---

## 4. Instruction Encoding Formats

All RISC-V instructions are 32 bits (fixed-length) in the base ISA. There are six encoding formats:

```
R-type:  [funct7 | rs2 | rs1 | funct3 | rd  | opcode]   (register-register)
I-type:  [   imm[11:0]  | rs1 | funct3 | rd  | opcode]   (register-immediate)
S-type:  [imm[11:5]|rs2 | rs1 | funct3 |imm[4:0]|opcode] (store)
B-type:  [imm bits | rs2| rs1 | funct3 |imm bits|opcode] (branch)
U-type:  [      imm[31:12]             | rd  | opcode]   (upper immediate)
J-type:  [      imm bits               | rd  | opcode]   (jump)
```

### 4.1 R-type Example: ADD

```
ADD x7, x5, x6
┌─────────┬──────┬──────┬────────┬──────┬─────────┐
│ 0000000 │ 00110│ 00101│  000   │ 00111│ 0110011 │
│ funct7  │ rs2  │ rs1  │ funct3 │  rd  │ opcode  │
└─────────┴──────┴──────┴────────┴──────┴─────────┘
  7 bits    5 bits 5 bits 3 bits  5 bits  7 bits = 32 bits total
```

### 4.2 Design Rationale

- **Fixed source register positions**: `rs1` and `rs2` are always in the same bit positions across all formats. This allows the register file to start reading before the instruction is fully decoded -- a pipeline optimization.
- **Sign extension**: Immediate values are always sign-extended from the most significant bit of the instruction (bit 31). The hardware sign-extension logic is trivially simple.

---

## 5. Standard Extensions

RISC-V's modularity comes from its extension system. Extensions are identified by single letters:

| Extension | Name | Description |
|-----------|------|-------------|
| **M** | Integer Multiply/Divide | `MUL`, `MULH`, `DIV`, `REM` and unsigned variants |
| **A** | Atomic | `LR.W/D` (load-reserved), `SC.W/D` (store-conditional), atomic swap/add/and/or |
| **F** | Single-Precision Float | 32 IEEE 754 float registers (f0-f31), `FADD.S`, `FMUL.S`, etc. |
| **D** | Double-Precision Float | Extends F registers to 64-bit, `FADD.D`, `FMUL.D`, etc. |
| **C** | Compressed | 16-bit encodings for the most common instructions (reduces code size ~25-30%) |
| **V** | Vector | Scalable vector processing (variable VLEN, not fixed like AVX/NEON) |

### 5.1 The "G" Shorthand

**RV32G** = RV32I + M + A + F + D (the "general-purpose" combination). Most application-class processors implement at least RV64GC (64-bit general-purpose with compressed instructions).

### 5.2 C Extension: Compressed Instructions

The C extension adds 16-bit encodings for frequent instructions. The processor intermixes 16-bit and 32-bit instructions in the same stream:

```asm
# 32-bit instruction (4 bytes)
add   x10, x11, x12     # Full R-type

# 16-bit compressed equivalent (2 bytes)
c.add x10, x12          # Same operation, half the size
```

Compressed instructions reduce code size by 25-30% -- comparable to ARM's Thumb-2. The trade-off: smaller immediate ranges and fewer accessible registers (only x8-x15 for some compressed formats).

### 5.3 V Extension: Vector Processing

Unlike x86 (SSE/AVX with fixed 128/256/512-bit vectors) or ARM (NEON with fixed 128-bit), RISC-V V uses **scalable vectors**:

```asm
# Set vector length for 32-bit elements
vsetvli t0, a0, e32, m1    # Process up to a0 elements, 32-bit each

# Vector load, add, store
vle32.v v1, (a1)            # Load vector from memory at a1
vle32.v v2, (a2)            # Load vector from memory at a2
vadd.vv v3, v1, v2          # v3 = v1 + v2 (element-wise)
vse32.v v3, (a3)            # Store result to memory at a3
```

The hardware determines how many elements fit per operation based on the physical vector register length (VLEN). The same binary runs on implementations with VLEN=128 or VLEN=1024 without recompilation.

---

## 6. Privilege Levels and CSRs

RISC-V defines three privilege levels for system software:

```
┌─────────────────────────────────────────┐
│ Level 3: Machine Mode (M-mode)          │ ← Firmware / Bootloader
│   - Highest privilege, always present   │
│   - Direct hardware access              │
├─────────────────────────────────────────┤
│ Level 1: Supervisor Mode (S-mode)       │ ← Operating System Kernel
│   - Virtual memory (Sv39, Sv48)         │
│   - Trap handling                       │
├─────────────────────────────────────────┤
│ Level 0: User Mode (U-mode)             │ ← Applications
│   - Restricted access                   │
│   - Uses ecall for system calls         │
└─────────────────────────────────────────┘
```

### 6.1 Control and Status Registers (CSRs)

CSRs control machine state:

| CSR | Purpose |
|-----|---------|
| `mstatus` | Machine status (interrupt enable, privilege mode) |
| `mtvec` | Machine trap vector base address |
| `mepc` | Machine exception program counter |
| `mcause` | Machine trap cause |
| `mhartid` | Hardware thread (hart) ID |
| `satp` | Supervisor address translation and protection |
| `cycle` | Cycle counter (read-only in U-mode) |

```asm
# Read cycle counter
csrr  t0, cycle          # t0 = current cycle count

# Enable interrupts
csrsi mstatus, 0x8       # Set MIE bit in mstatus
```

---

## 7. RISC-V vs x86 vs ARM

### 7.1 Architecture Comparison

| Feature | RISC-V | x86-64 | ARMv8 (AArch64) |
|---------|--------|--------|-----------------|
| **ISA Type** | RISC, open | CISC, proprietary | RISC, proprietary |
| **License** | Free | Intel/AMD only | Royalty-based |
| **Instruction Length** | 32-bit (16 with C ext) | Variable (1-15 bytes) | Fixed 32-bit |
| **GP Registers** | 32 (x0 = zero) | 16 | 31 (x0-x30) + xzr |
| **Addressing Modes** | Base + offset only | Many (complex) | Base + offset, some complex |
| **Condition Codes** | None (compare-and-branch) | FLAGS register | FLAGS (NZCV) register |
| **Multiply in base** | No (M extension) | Yes | Yes |
| **SIMD/Vector** | V extension (scalable) | SSE/AVX (fixed width) | NEON (128-bit) / SVE (scalable) |
| **Endianness** | Little (default) | Little | Bi-endian |
| **Privilege Levels** | 3 (M/S/U) | 4 (Ring 0-3) | 4 (EL0-EL3) |

### 7.2 No Condition Flags

RISC-V eliminates the FLAGS/condition-code register found in x86 and ARM. Instead, branches directly compare two registers:

```asm
# x86: compare then branch (two instructions, implicit state)
cmp   eax, ebx
je    target

# ARM: compare then branch (two instructions, modifies NZCV)
cmp   w0, w1
b.eq  target

# RISC-V: fused compare-and-branch (one instruction, no side effects)
beq   x10, x11, target
```

This simplifies out-of-order execution because branches have no hidden dependencies on flag registers.

### 7.3 Code Size Comparison

For the same "Hello, World" program:

| ISA | Binary Size | Instruction Count |
|-----|------------|-------------------|
| RV64GC | ~4.2 KB | 15 instructions |
| AArch64 | ~4.0 KB | 14 instructions |
| x86-64 | ~4.5 KB | 12 instructions (but variable-length) |

With the C (compressed) extension, RISC-V achieves code density comparable to ARM Thumb-2 and better than fixed-32-bit ARM.

---

## 8. RISC-V Ecosystem

### 8.1 Hardware

| Category | Examples |
|----------|---------|
| **MCU** | ESP32-C3 (WiFi+BLE), GD32V, CH32V |
| **Application** | SiFive U74 (Linux-capable), StarFive JH7110 |
| **HPC** | Sophon SG2042 (64 cores), Ventana Veyron |
| **Development Boards** | HiFive Unmatched, VisionFive 2, Milk-V Pioneer |

### 8.2 Software

- **Compilers**: GCC 13+, LLVM/Clang 17+ (full support including V extension)
- **OS**: Linux (mainline kernel support since 4.15), FreeBSD, Zephyr RTOS, FreeRTOS
- **Simulators**: QEMU, Spike (reference), gem5
- **Toolchains**: `riscv64-unknown-elf-gcc` (bare-metal), `riscv64-linux-gnu-gcc` (Linux)

### 8.3 Simulation with QEMU

```bash
# Install RISC-V QEMU (Ubuntu/Debian)
sudo apt install qemu-system-riscv64

# Compile bare-metal RISC-V program
riscv64-unknown-elf-gcc -march=rv64gc -o hello hello.c -nostdlib

# Run in QEMU
qemu-system-riscv64 -machine virt -nographic -bios none -kernel hello
```

---

## 9. Practice Problems

### Problem 1: Instruction Encoding

Encode the following instruction as a 32-bit binary value. Identify the format type (R/I/S/B/U/J) and fill in each field:

```asm
addi x10, x5, -3
```

### Problem 2: Assembly Tracing

Trace the register values after each instruction:

```asm
    addi  x5, x0, 10      # x5 = ?
    addi  x6, x0, 3       # x6 = ?
    add   x7, x5, x6      # x7 = ?
    sub   x8, x5, x6      # x8 = ?
    slli  x9, x6, 2       # x9 = ? (shift left logical by 2)
    and   x10, x5, x6     # x10 = ?
    or    x11, x5, x6     # x11 = ?
    slt   x12, x6, x5     # x12 = ? (set if x6 < x5)
```

### Problem 3: ISA Comparison

Write a function `int max(int a, int b)` in:
1. RISC-V RV32I assembly (use `bge` / `blt`)
2. x86-64 assembly (use `cmp` / `cmovge`)
3. ARMv8 AArch64 assembly (use `cmp` / `csel`)

Compare the number of instructions and identify which ISA feature (condition codes, conditional moves, predication) affects the implementation.

### Problem 4: Extension Analysis

A RISC-V processor is labeled "RV64IMAFDC". List each component and explain:
1. What each letter means
2. What instructions each extension adds
3. What the shorthand "RV64GC" means and why it is equivalent

### Problem 5: Pipeline Hazard

Consider the following RISC-V code on a 5-stage pipeline (IF-ID-EX-MEM-WB):

```asm
    lw    x5, 0(x10)       # Load word
    add   x6, x5, x7      # Uses x5 immediately
    sub   x8, x5, x9      # Also uses x5
```

1. Identify the data hazard type (RAW/WAR/WAW)
2. How many stall cycles are needed without forwarding?
3. How many with forwarding (EX-EX and MEM-EX)?
4. Can the compiler eliminate the stall? If so, how?

---

*End of Lesson 19*
