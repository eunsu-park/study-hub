# Computer Architecture Study Guide

## Introduction

This folder contains materials for systematically learning Computer Architecture. You'll understand how computers work, from data representation to CPU architecture, memory systems, and parallel processing.

**Target Audience**: Developers with basic programming knowledge, those learning CS fundamentals

---

## Learning Roadmap

```
[Basics]                  [Intermediate]            [Advanced]
  │                         │                         │
  ▼                         ▼                         ▼
Computer Overview ───▶ Instruction Set ────▶ Pipelining
  │                         │                         │
  ▼                         ▼                         ▼
Data Representation ─▶ Control Unit ───────▶ Cache Memory
  │                         │                         │
  ▼                         ▼                         ▼
Logic Gates ─────────▶ CPU Architecture ───▶ Virtual Memory
  │                                                   │
  ▼                                                   ▼
Sequential Logic ───────────────────────────▶ Parallel/Multicore
```

---

## Prerequisites

- Programming basics (variables, control flow, functions)
- Basic mathematics (binary numbers, logical operations)
- At least one language: C or Python

---

## File List

### Basic Concepts (01-05)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [01_Computer_System_Overview.md](./01_Computer_System_Overview.md) | ⭐ | Computer history, Von Neumann architecture, hardware components |
| [02_Data_Representation_Basics.md](./02_Data_Representation_Basics.md) | ⭐ | Binary, octal, hexadecimal, base conversion |
| [03_Integer_Float_Representation.md](./03_Integer_Float_Representation.md) | ⭐⭐ | Two's complement, IEEE 754 floating-point |
| [04_Logic_Gates.md](./04_Logic_Gates.md) | ⭐ | AND, OR, NOT, Boolean algebra |
| [05_Combinational_Logic.md](./05_Combinational_Logic.md) | ⭐⭐ | Adders, multiplexers, decoders |

### CPU Architecture (06-10)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [06_Sequential_Logic.md](./06_Sequential_Logic.md) | ⭐⭐ | Flip-flops, registers, counters |
| [07_CPU_Architecture_Basics.md](./07_CPU_Architecture_Basics.md) | ⭐⭐ | ALU, register file, datapath |
| [08_Control_Unit.md](./08_Control_Unit.md) | ⭐⭐⭐ | Hardwired/microprogrammed control |
| [09_Instruction_Set_Architecture.md](./09_Instruction_Set_Architecture.md) | ⭐⭐⭐ | CISC vs RISC, addressing modes |
| [10_Assembly_Language_Basics.md](./10_Assembly_Language_Basics.md) | ⭐⭐⭐ | x86/ARM basics, basic instructions |

### Performance Enhancement Techniques (11-13)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [11_Pipelining.md](./11_Pipelining.md) | ⭐⭐⭐ | Pipeline stages, hazards, forwarding |
| [12_Branch_Prediction.md](./12_Branch_Prediction.md) | ⭐⭐⭐ | Static/dynamic branch prediction, BTB |
| [13_Superscalar_Out_of_Order.md](./13_Superscalar_Out_of_Order.md) | ⭐⭐⭐⭐ | ILP, register renaming |

### Memory Systems (14-16)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [14_Memory_Hierarchy.md](./14_Memory_Hierarchy.md) | ⭐⭐ | Locality, SRAM/DRAM, memory hierarchy |
| [15_Cache_Memory.md](./15_Cache_Memory.md) | ⭐⭐⭐ | Direct/associative/set-associative mapping, replacement policies |
| [16_Virtual_Memory.md](./16_Virtual_Memory.md) | ⭐⭐⭐⭐ | Page tables, TLB, page replacement |

### I/O and Parallel Processing (17-18)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [17_IO_Systems.md](./17_IO_Systems.md) | ⭐⭐⭐ | Interrupts, DMA, buses |
| [18_Parallel_Processing_Multicore.md](./18_Parallel_Processing_Multicore.md) | ⭐⭐⭐⭐ | SIMD/MIMD, cache coherence, Amdahl's Law |

### Modern Topics (19-20)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [19_RISC_V_Architecture.md](./19_RISC_V_Architecture.md) | ⭐⭐⭐ | RISC-V ISA, open-source hardware, extensions |
| [20_Power_Management.md](./20_Power_Management.md) | ⭐⭐⭐ | Power states, DVFS, thermal management, energy efficiency |

---

## Recommended Study Order

### Phase 1: Basic Concepts (1 week)
```
01_Computer_System_Overview → 02_Data_Representation_Basics → 03_Integer_Float_Representation
```

### Phase 2: Digital Logic (1 week)
```
04_Logic_Gates → 05_Combinational_Logic → 06_Sequential_Logic
```

### Phase 3: CPU Architecture (2 weeks)
```
07_CPU_Architecture_Basics → 08_Control_Unit → 09_Instruction_Set_Architecture → 10_Assembly_Language_Basics
```

### Phase 4: Performance Enhancement (1-2 weeks)
```
11_Pipelining → 12_Branch_Prediction → 13_Superscalar_Out_of_Order
```

### Phase 5: Memory Systems (1-2 weeks)
```
14_Memory_Hierarchy → 15_Cache_Memory → 16_Virtual_Memory
```

### Phase 6: I/O and Parallel Processing (1 week)
```
17_IO_Systems → 18_Parallel_Processing_Multicore
```

---

## Practice Environment

### Simulators

```bash
# MARS (MIPS Simulator)
# https://courses.missouristate.edu/kenvollmar/mars/

# Logisim (Digital Circuit Simulator)
# https://www.cburch.com/logisim/

# CPU Simulator
# https://cpuvisualsimulator.github.io/
```

### Assembly Practice

```bash
# x86 (Linux)
nasm -f elf64 hello.asm -o hello.o
ld hello.o -o hello

# GCC Assembly Output
gcc -S -O0 program.c -o program.s
```

---

## Performance Quick Reference

| Component | Typical Latency |
|-----------|----------------|
| Register access | ~1 cycle |
| L1 cache | ~4 cycles |
| L2 cache | ~10 cycles |
| L3 cache | ~40 cycles |
| Main memory | ~100+ cycles |
| SSD | ~10,000+ cycles |
| HDD | ~10,000,000+ cycles |

---

## Related Resources

### Links to Other Folders

| Folder | Related Content |
|--------|----------------|
| [C_Programming/](../C_Programming/00_Overview.md) | Pointers, memory management |
| [Algorithm/](../Algorithm/00_Overview.md) | Complexity analysis, cache optimization |
| [Linux/](../Linux/00_Overview.md) | Processes, memory management |

### External Resources

- [Computer Organization and Design (Patterson & Hennessy)](https://www.elsevier.com/books/computer-organization-and-design/)
- [Nand2Tetris](https://www.nand2tetris.org/)
- [CPU Visualization](https://www.youtube.com/watch?v=cNN_tTXABUA)

---

## Study Tips

1. **Use Simulators**: Implement digital circuits directly in Logisim
2. **Assembly Practice**: Write simple programs in assembly language
3. **Cache Analysis**: Analyze cache misses using perf or cachegrind
4. **Step-by-Step Understanding**: Always solve practice problems in each lesson
5. **Visualization**: Understand pipelines, cache operations through diagrams
