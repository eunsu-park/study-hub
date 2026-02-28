# Parallel Processing and Multicore

**Previous**: [I/O Systems](./17_IO_Systems.md) | **Next**: [RISC-V Architecture](./19_RISC_V_Architecture.md)

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: CPU architecture, cache memory, memory hierarchy

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why clock frequency scaling hit a power wall
2. Describe Flynn's taxonomy (SISD, SIMD, MISD, MIMD)
3. Explain Amdahl's law and calculate theoretical speedup limits
4. Compare shared-memory and distributed-memory multiprocessor architectures
5. Describe cache coherence protocols (MESI) and why they are necessary
6. Explain the role of GPUs and SIMT execution in modern computing

---

For decades, CPUs got faster every year by increasing clock frequency. That era ended around 2005 when power consumption hit a wall. The industry's answer was multicore -- putting multiple processors on a single chip. Understanding parallel processing explains why a 16-core processor is not 16x faster than a single core, why cache coherence is hard, and why GPUs dominate machine learning workloads.

## Table of Contents

1. [The Need for Parallel Processing](#1-the-need-for-parallel-processing)
2. [Flynn's Taxonomy](#2-flynns-taxonomy)
3. [Multiprocessor and Multicore](#3-multiprocessor-and-multicore)
4. [Cache Coherence Problem](#4-cache-coherence-problem)
5. [Snooping Protocol (MESI)](#5-snooping-protocol-mesi)
6. [Amdahl's Law and Gustafson's Law](#6-amdahls-law-and-gustafsons-law)
7. [Synchronization and Locks](#7-synchronization-and-locks)
8. [GPU and Parallel Computing](#8-gpu-and-parallel-computing)
9. [Practice Problems](#9-practice-problems)

---

## 1. The Need for Parallel Processing

### 1.1 Single Core Limitations

```
┌─────────────────────────────────────────────────────────────┐
│              CPU Clock Speed Evolution                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Clock                                                      │
│  (GHz)                                                      │
│    │                                    ●──────● (Plateau)  │
│  5 │                               ●────●                   │
│    │                          ●────                         │
│  4 │                     ●────                              │
│    │                ●────                                   │
│  3 │           ●────                                        │
│    │      ●────                                             │
│  2 │  ●────                                                 │
│    │ ●                                                      │
│  1 │●                                                       │
│    │                                                        │
│    └────────────────────────────────────────────────────    │
│     1995    2000    2005    2010    2015    2020            │
│                                                             │
│  Clock speed plateau after 2005:                           │
│  - Power Wall                                              │
│  - Heat dissipation issues                                 │
│  - Memory Wall                                             │
│  - ILP Wall                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Moore's Law and Dennard Scaling

```
Moore's Law:
- Transistor count doubles every 2 years
- Still valid (slowing down)

Dennard Scaling:
- Transistor size reduction → constant power density
- Broke down around 2006

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  After end of Dennard Scaling:                             │
│                                                             │
│  Transistor count ↑  +  Clock speed plateau  →  How to use?│
│                                                             │
│  Solution: Multicore                                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │   Before 2005:        After 2005:                   │   │
│  │                                                     │   │
│  │   ┌─────────────┐    ┌─────┐ ┌─────┐ ┌─────┐      │   │
│  │   │             │    │Core1│ │Core2│ │Core3│ ...  │   │
│  │   │   Single    │    │     │ │     │ │     │      │   │
│  │   │  High-Perf  │    └─────┘ └─────┘ └─────┘      │   │
│  │   │    Core     │                                  │   │
│  │   └─────────────┘    Multiple efficient cores      │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Benefits of Parallel Processing

```
Performance Improvement:
- Process multiple tasks simultaneously
- Divide large problems into smaller parallel solutions

Energy Efficiency:
- Multiple cores at lower clock more efficient than single core at high clock
- Power ∝ Voltage² × Frequency

Reliability:
- Continue operation with other cores if one core fails
- Fault Tolerance

Scalability:
- Performance scaling through increased core count
- Vertical scaling (add cores) + Horizontal scaling (add nodes)
```

---

## 2. Flynn's Taxonomy

### 2.1 Classification System

```
┌─────────────────────────────────────────────────────────────┐
│                   Flynn's Taxonomy                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│              │     Single Data     │    Multiple Data       │
│              │        (SD)         │        (MD)            │
│  ────────────┼─────────────────────┼────────────────────────│
│              │                     │                        │
│   Single     │       SISD          │       SIMD            │
│ Instruction  │  (Single Instruction│  (Single Instruction  │
│     (SI)     │   Single Data)      │   Multiple Data)      │
│              │                     │                        │
│  ────────────┼─────────────────────┼────────────────────────│
│              │                     │                        │
│  Multiple    │       MISD          │       MIMD            │
│ Instruction  │  (Multiple Instr.   │  (Multiple Instr.     │
│     (MI)     │   Single Data)      │   Multiple Data)      │
│              │                     │                        │
│              │   (Rarely used)     │                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 SISD (Single Instruction, Single Data)

```
Traditional von Neumann computer:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     Instruction Stream          Data Stream                 │
│           │                         │                       │
│           ▼                         ▼                       │
│     ┌───────────┐             ┌───────────┐                │
│     │    I1     │             │    D1     │                │
│     │    I2     │             │    D2     │                │
│     │    I3     │             │    D3     │                │
│     │    ...    │             │    ...    │                │
│     └─────┬─────┘             └─────┬─────┘                │
│           │                         │                       │
│           └───────────┬─────────────┘                       │
│                       │                                     │
│                       ▼                                     │
│               ┌───────────────┐                            │
│               │      CPU      │                            │
│               │ (Single Core) │                            │
│               └───────────────┘                            │
│                                                             │
│  Examples: Early microprocessors, embedded systems          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 SIMD (Single Instruction, Multiple Data)

```
Same operation applied to multiple data simultaneously:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     Single Instruction              Multiple Data           │
│           │                    ┌──────┬──────┬──────┐      │
│           │                    │  D1  │  D2  │  D3  │      │
│           ▼                    └──┬───┴──┬───┴──┬───┘      │
│     ┌───────────┐                 │      │      │          │
│     │   ADD     │                 ▼      ▼      ▼          │
│     └─────┬─────┘            ┌────────────────────────┐    │
│           │                  │     Processing Units    │    │
│           │                  │   ┌────┐┌────┐┌────┐   │    │
│           └─────────────────▶│   │ PU1││ PU2││ PU3│   │    │
│                              │   └────┘└────┘└────┘   │    │
│                              └──────────┬─────────────┘    │
│                                         │                   │
│                              ┌──────────┴──────────┐       │
│                              │  R1  │  R2  │  R3  │       │
│                              └──────┴──────┴──────┘       │
│                                                             │
│  Examples:                                                  │
│  - Intel SSE, AVX (256/512-bit vectors)                    │
│  - GPU warps/waves                                         │
│  - Image processing, scientific computing                   │
│                                                             │
│  Code example (AVX):                                        │
│  __m256 a = _mm256_load_ps(arr1);                          │
│  __m256 b = _mm256_load_ps(arr2);                          │
│  __m256 c = _mm256_add_ps(a, b);  // 8 floats added at once│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 MISD (Multiple Instruction, Single Data)

```
Multiple instructions processing same data stream:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     Multiple Instructions           Single Data             │
│     ┌────────────────┐                  │                   │
│     │ I1: Encrypt    │                  │                   │
│     │ I2: Compress   │                  ▼                   │
│     │ I3: Checksum   │             ┌─────────┐             │
│     └───────┬────────┘             │  Data   │             │
│             │                      └────┬────┘             │
│             │                           │                   │
│             ▼                           │                   │
│      ┌────────────┐                     │                   │
│      │ Pipeline   │◀────────────────────┘                   │
│      │ of Units   │                                         │
│      └────────────┘                                         │
│                                                             │
│  Real-world usage:                                          │
│  - Systolic arrays (some)                                  │
│  - Fault-tolerant systems (same computation multiple times) │
│  - Very rarely used                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.5 MIMD (Multiple Instruction, Multiple Data)

```
Most common parallel computer form:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     Multiple Instructions           Multiple Data           │
│     ┌────────────────┐         ┌────────────────────┐      │
│     │ I1: func_A()   │         │ D1, D2, D3, D4     │      │
│     │ I2: func_B()   │         │ D5, D6, D7, D8     │      │
│     │ I3: func_C()   │         │ ...               │      │
│     │ I4: func_D()   │         │                   │      │
│     └───────┬────────┘         └─────────┬─────────┘      │
│             │                            │                  │
│             ▼                            ▼                  │
│     ┌───────────────────────────────────────────────────┐  │
│     │               Multiple Processors                  │  │
│     │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │
│     │  │  CPU 1  │ │  CPU 2  │ │  CPU 3  │ │  CPU 4  │ │  │
│     │  │ func_A()│ │ func_B()│ │ func_C()│ │ func_D()│ │  │
│     │  │   D1    │ │   D5    │ │   D2    │ │   D8    │ │  │
│     │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │  │
│     └───────────────────────────────────────────────────┘  │
│                                                             │
│  Examples:                                                  │
│  - Multicore processors                                    │
│  - Multiprocessor servers                                  │
│  - Clusters, supercomputers                                │
│                                                             │
│  MIMD Classification:                                       │
│  - Shared Memory: SMP, NUMA                                │
│  - Distributed Memory: Clusters                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Multiprocessor and Multicore

### 3.1 SMP (Symmetric Multi-Processing)

```
┌─────────────────────────────────────────────────────────────┐
│                  SMP (Symmetric Multi-Processing)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│     │  CPU 0  │ │  CPU 1  │ │  CPU 2  │ │  CPU 3  │       │
│     │┌───────┐│ │┌───────┐│ │┌───────┐│ │┌───────┐│       │
│     ││ Cache ││ ││ Cache ││ ││ Cache ││ ││ Cache ││       │
│     │└───────┘│ │└───────┘│ │└───────┘│ │└───────┘│       │
│     └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
│          │           │           │           │             │
│          └───────────┴─────┬─────┴───────────┘             │
│                            │                                │
│                      System Bus                            │
│                            │                                │
│     ┌──────────────────────┴──────────────────────┐        │
│     │                                             │        │
│     │              Shared Memory                  │        │
│     │            (Equal access for all CPUs)      │        │
│     │                                             │        │
│     └─────────────────────────────────────────────┘        │
│                                                             │
│  Characteristics:                                           │
│  - All processors have equal memory access (UMA)           │
│  - Any CPU can run same code                               │
│  - Limited scalability (bus contention)                    │
│  - Typically 2-8 CPUs                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 NUMA (Non-Uniform Memory Access)

```
┌─────────────────────────────────────────────────────────────┐
│                    NUMA Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│          Node 0                        Node 1               │
│     ┌──────────────────┐          ┌──────────────────┐     │
│     │  CPU0    CPU1    │          │  CPU2    CPU3    │     │
│     │  ┌───┐   ┌───┐   │          │  ┌───┐   ┌───┐   │     │
│     │  │ $ │   │ $ │   │          │  │ $ │   │ $ │   │     │
│     │  └─┬─┘   └─┬─┘   │          │  └─┬─┘   └─┬─┘   │     │
│     │    └───┬───┘     │          │    └───┬───┘     │     │
│     │        │         │          │        │         │     │
│     │  ┌─────┴─────┐   │          │  ┌─────┴─────┐   │     │
│     │  │ Local Mem │   │◀════════▶│  │ Local Mem │   │     │
│     │  │  (Fast)   │   │Interconn│  │  (Fast)   │   │     │
│     │  └───────────┘   │          │  └───────────┘   │     │
│     └──────────────────┘          └──────────────────┘     │
│                                                             │
│  Memory access time:                                        │
│  ┌────────────────────────────────────────────────────────┐│
│  │ Local Memory (same node):     ~100 cycles              ││
│  │ Remote Memory (other node):   ~300 cycles (3x slower)  ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  Characteristics:                                           │
│  - Local memory access faster than remote memory           │
│  - Excellent scalability (hundreds of CPUs possible)       │
│  - Requires NUMA-aware programming                         │
│  - Standard architecture for modern servers                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Multicore Processor

```
┌─────────────────────────────────────────────────────────────┐
│                  Modern Multicore CPU                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                    CPU Die                             │ │
│  │  ┌─────────────────────────────────────────────────┐  │ │
│  │  │    Core 0           Core 1          Core 2      │  │ │
│  │  │  ┌─────────┐      ┌─────────┐     ┌─────────┐  │  │ │
│  │  │  │L1-I│L1-D│      │L1-I│L1-D│     │L1-I│L1-D│  │  │ │
│  │  │  └────┴────┘      └────┴────┘     └────┴────┘  │  │ │
│  │  │  ┌─────────┐      ┌─────────┐     ┌─────────┐  │  │ │
│  │  │  │   L2    │      │   L2    │     │   L2    │  │  │ │
│  │  │  └─────────┘      └─────────┘     └─────────┘  │  │ │
│  │  └─────────────────────────────────────────────────┘  │ │
│  │                          │                             │ │
│  │  ┌───────────────────────┴───────────────────────┐    │ │
│  │  │                  Shared L3 Cache               │    │ │
│  │  │                   (8-64 MB)                    │    │ │
│  │  └───────────────────────┬───────────────────────┘    │ │
│  │                          │                             │ │
│  │  ┌───────────────────────┴───────────────────────┐    │ │
│  │  │            Memory Controller                   │    │ │
│  │  │            + PCIe Controller                   │    │ │
│  │  └───────────────────────────────────────────────┘    │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Advantages:                                                │
│  - Fast inter-core communication (on-chip)                 │
│  - Power efficient                                         │
│  - Efficient data sharing through shared cache             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Hyper-Threading (SMT)

```
┌─────────────────────────────────────────────────────────────┐
│           Simultaneous Multi-Threading (SMT)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Execute multiple threads on single physical core:          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Physical Core                      │   │
│  │  ┌─────────────────────────────────────────────────┐│   │
│  │  │ Thread 0 State │ Thread 1 State                 ││   │
│  │  │  ┌───────────┐ │ ┌───────────┐                  ││   │
│  │  │  │Registers  │ │ │Registers  │  ← Duplicated   ││   │
│  │  │  │PC, Stack  │ │ │PC, Stack  │                  ││   │
│  │  │  └───────────┘ │ └───────────┘                  ││   │
│  │  └─────────────────────────────────────────────────┘│   │
│  │                                                      │   │
│  │  ┌─────────────────────────────────────────────────┐│   │
│  │  │              Shared Resources                   ││   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐          ││   │
│  │  │  │  ALU    │ │  Cache  │ │ Branch  │  ← Shared││   │
│  │  │  │         │ │         │ │Predictor│          ││   │
│  │  │  └─────────┘ └─────────┘ └─────────┘          ││   │
│  │  └─────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Operation:                                                 │
│  - When Thread 0 waits for memory → Thread 1 executes      │
│  - Improves execution unit utilization                     │
│  - Typically 15-30% performance improvement                │
│                                                             │
│  OS view:                                                   │
│  - 4-core 8-thread = recognized as 8 logical CPUs          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Cache Coherence Problem

### 4.1 What is Cache Coherence?

```
Problem scenario:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     Core 0                            Core 1                │
│     Cache                             Cache                 │
│   ┌─────────┐                       ┌─────────┐            │
│   │ X = 10  │                       │ X = 10  │            │
│   └─────────┘                       └─────────┘            │
│        │                                 │                  │
│        └─────────────┬───────────────────┘                  │
│                      │                                      │
│               ┌──────┴──────┐                              │
│               │ Main Memory │                              │
│               │   X = 10    │                              │
│               └─────────────┘                              │
│                                                             │
│  1. Initial state: X = 10 (all same)                       │
│                                                             │
│  2. Core 0 modifies X = 20:                                │
│                                                             │
│     Core 0                            Core 1                │
│     Cache                             Cache                 │
│   ┌─────────┐                       ┌─────────┐            │
│   │ X = 20  │ ← Modified            │ X = 10  │ ← Stale!   │
│   └─────────┘                       └─────────┘            │
│                                                             │
│  3. What value does Core 1 read for X?                     │
│     - 10 (its own cache) → Stale value! (coherence violation)│
│     - 20 (Core 0's value) → Coherence maintained           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Coherence Definition

```
Cache Coherence conditions:

1. Program order preservation:
   - Write followed by read on same processor returns written value

2. Consistent read values:
   - Writes by other processors eventually visible to all processors

3. Write serialization:
   - Writes to same address appear in same order to all processors

┌─────────────────────────────────────────────────────────────┐
│  Example:                                                   │
│                                                             │
│  Initial: X = 0                                             │
│                                                             │
│  Core 0: X = 1                                              │
│  Core 1: X = 2                                              │
│                                                             │
│  Order of X values seen must be same for all processors:    │
│  - All see 0 → 1 → 2 order, OR                             │
│  - All see 0 → 2 → 1 order                                 │
│                                                             │
│  Some processors seeing 1 → 2 while others see 2 → 1 is invalid│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Coherence Protocol Overview

```
Coherence maintenance methods:

1. Snooping Protocol:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  - Suitable for bus-based systems                          │
│  - Each cache monitors bus traffic (snooping)              │
│  - Takes appropriate action when relevant address detected  │
│  - MESI, MOESI, etc.                                       │
│                                                             │
│     ┌─────┐     ┌─────┐     ┌─────┐                        │
│     │Cache│     │Cache│     │Cache│  ← All monitor bus     │
│     └──┬──┘     └──┬──┘     └──┬──┘                        │
│        │           │           │                            │
│     ═══╪═══════════╪═══════════╪═════  (Shared Bus)        │
│                    │                                        │
│             ┌──────┴──────┐                                │
│             │   Memory    │                                │
│             └─────────────┘                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

2. Directory Protocol:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  - Good scalability (suitable for NUMA)                    │
│  - Central directory tracks cache states                   │
│  - Send messages only to relevant caches                   │
│                                                             │
│     ┌─────┐           ┌─────┐           ┌─────┐           │
│     │Cache│           │Cache│           │Cache│           │
│     └──┬──┘           └──┬──┘           └──┬──┘           │
│        │                 │                 │               │
│        └────────┬────────┴────────┬────────┘               │
│                 │                 │                         │
│           ┌─────┴─────┐   ┌──────┴──────┐                  │
│           │ Directory │   │   Memory    │                  │
│           │(State Trac)│   │             │                  │
│           └───────────┘   └─────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Snooping Protocol (MESI)

### 5.1 MESI States

```
┌─────────────────────────────────────────────────────────────┐
│                    MESI Protocol States                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  M (Modified):                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - This cache only has copy                          │   │
│  │ - Modified (inconsistent with memory)               │   │
│  │ - Write-back needed on other cache access           │   │
│  │ - Write allowed                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  E (Exclusive):                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - This cache only has copy                          │   │
│  │ - Not modified (consistent with memory)             │   │
│  │ - Transition to M on write (no invalidation needed) │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  S (Shared):                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - Multiple caches may have copy                     │   │
│  │ - Consistent with memory                            │   │
│  │ - On write, invalidate other caches → M state       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  I (Invalid):                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - No valid data                                     │   │
│  │ - Cache line empty                                  │   │
│  │ - Cache miss on access                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 MESI State Transitions

```
┌─────────────────────────────────────────────────────────────┐
│                   MESI State Transition Diagram              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      ┌─────────┐                            │
│           Read miss  │         │  Read miss                 │
│           (exclusive)│    I    │  (shared)                  │
│             ┌────────│ Invalid │────────┐                   │
│             │        │         │        │                   │
│             │        └────┬────┘        │                   │
│             │             │             │                   │
│             │        Write│             │                   │
│             │        miss │             │                   │
│             │             │             │                   │
│             ▼             │             ▼                   │
│       ┌──────────┐        │      ┌──────────┐              │
│       │    E     │        │      │    S     │              │
│       │Exclusive │        │      │ Shared   │              │
│       └────┬─────┘        │      └────┬─────┘              │
│            │              │           │                     │
│       Local│              │      Local│                     │
│       Write│              │      Write│                     │
│            │              │           │                     │
│            ▼              ▼           ▼                     │
│       ┌─────────────────────────────────┐                  │
│       │            M                    │                  │
│       │         Modified                │                  │
│       └─────────────────────────────────┘                  │
│                                                             │
│  Key transitions:                                           │
│  - I → E: Read miss, not in other caches                   │
│  - I → S: Read miss, exists in other caches (Shared state) │
│  - E → M: Local write (no invalidation broadcast needed)   │
│  - S → M: Local write + invalidate other caches            │
│  - M/E/S → I: Invalidated by other cache's write           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 MESI Operation Example

```
┌─────────────────────────────────────────────────────────────┐
│              MESI Protocol Operation Example                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Initial: Variable X not in any cache (only in memory X=0)  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 1: Core 0 reads X                              │   │
│  │                                                     │   │
│  │   Core 0 Cache     Core 1 Cache     Memory          │   │
│  │   ┌─────────┐      ┌─────────┐      ┌─────────┐    │   │
│  │   │ X=0 (E) │      │ X (I)   │      │ X = 0   │    │   │
│  │   └─────────┘      └─────────┘      └─────────┘    │   │
│  │   Exclusive (not in other caches)                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 2: Core 1 reads X                              │   │
│  │                                                     │   │
│  │   Core 0 Cache     Core 1 Cache     Memory          │   │
│  │   ┌─────────┐      ┌─────────┐      ┌─────────┐    │   │
│  │   │ X=0 (S) │      │ X=0 (S) │      │ X = 0   │    │   │
│  │   └─────────┘      └─────────┘      └─────────┘    │   │
│  │   E→S transition (read by other cache)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 3: Core 0 writes X = 10                        │   │
│  │                                                     │   │
│  │   Core 0: Broadcast invalidation message            │   │
│  │   Core 1: Change X to Invalid                       │   │
│  │                                                     │   │
│  │   Core 0 Cache     Core 1 Cache     Memory          │   │
│  │   ┌─────────┐      ┌─────────┐      ┌─────────┐    │   │
│  │   │ X=10(M) │      │ X (I)   │      │ X = 0   │    │   │
│  │   └─────────┘      └─────────┘      └─────────┘    │   │
│  │   Modified (inconsistent with memory)               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 4: Core 1 reads X                              │   │
│  │                                                     │   │
│  │   Core 1: Read miss, Core 0 provides data           │   │
│  │   Core 0: Write-back to memory, transition to S     │   │
│  │                                                     │   │
│  │   Core 0 Cache     Core 1 Cache     Memory          │   │
│  │   ┌─────────┐      ┌─────────┐      ┌─────────┐    │   │
│  │   │ X=10(S) │      │ X=10(S) │      │ X = 10  │    │   │
│  │   └─────────┘      └─────────┘      └─────────┘    │   │
│  │   M→S transition, memory updated                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 MOESI Protocol

```
MOESI = MESI + Owner state:

O (Owner):
- Sole owner of modified data
- Other caches may have Shared copies
- Inconsistent with memory (Owner has latest)
- Owner responds to other cache requests

┌─────────────────────────────────────────────────────────────┐
│  MOESI advantages:                                          │
│  - Increased cache-to-cache transfer efficiency            │
│  - Can delay write-back                                    │
│  - Mainly used in AMD processors                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Amdahl's Law and Gustafson's Law

### 6.1 Amdahl's Law

```
Law showing limits of parallelization:

┌─────────────────────────────────────────────────────────────┐
│                     Amdahl's Law                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                        1                                    │
│  Speedup = ─────────────────────────                        │
│            (1 - P) + P/N                                    │
│                                                             │
│  P: Parallelizable fraction                                 │
│  N: Number of processors                                    │
│                                                             │
│  Example: P = 90% (90% parallelizable)                     │
│                                                             │
│  N=2:   Speedup = 1/(0.1 + 0.45) = 1.82x                   │
│  N=4:   Speedup = 1/(0.1 + 0.225) = 3.08x                  │
│  N=8:   Speedup = 1/(0.1 + 0.1125) = 4.71x                 │
│  N=∞:   Speedup = 1/0.1 = 10x  ← Maximum limit!            │
│                                                             │
│  Even with 90% parallelization, max 10x speedup possible    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Speedup                                              │   │
│  │    │                              _______ P=99%     │   │
│  │ 100│                        _____/                  │   │
│  │    │                   ____/                        │   │
│  │ 80 │              ____/                             │   │
│  │    │         ____/              _______ P=95%       │   │
│  │ 60 │    ____/              ____/                    │   │
│  │    │___/              ____/                         │   │
│  │ 40 │             ____/          _______ P=90%       │   │
│  │    │        ____/          ____/                    │   │
│  │ 20 │   ____/          ____/                         │   │
│  │    │__/          ____/_______________________ P=75% │   │
│  │    │        ____/___________________________        │   │
│  │    └────────────────────────────────────────────    │   │
│  │    1    10   100  1000  10000  Number of processors │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Gustafson's Law

```
More parallelization possible by scaling problem size:

┌─────────────────────────────────────────────────────────────┐
│                    Gustafson's Law                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Scaled Speedup = N + (1 - N) × S                          │
│                                                             │
│  Or:                                                        │
│                                                             │
│  Speedup = N - S × (N - 1)                                 │
│                                                             │
│  N: Number of processors                                    │
│  S: Sequential fraction (fixed time)                        │
│                                                             │
│  Key idea:                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  Amdahl: "Fixed problem size, add processors"      │   │
│  │          → Sequential part becomes bottleneck       │   │
│  │                                                     │   │
│  │  Gustafson: "Fixed time, scale problem size"       │   │
│  │             → Solve larger problem in same time     │   │
│  │             → Parallel portion fraction increases   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Example:                                                   │
│  - Run simulation in 1 hour                                │
│  - Adding processors allows more detailed simulation        │
│  - Sequential part (initialization, etc.) stays constant    │
│    parallel computation volume increases                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Actual Parallelization Efficiency

```
┌─────────────────────────────────────────────────────────────┐
│              Parallelization Efficiency Calculation          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Efficiency = Speedup / N                                  │
│                                                             │
│  Ideal: Efficiency = 1 (100%)                              │
│  Realistic: Less than 1 due to overhead                    │
│                                                             │
│  Overhead factors:                                          │
│  - Communication time (inter-processor data transfer)       │
│  - Synchronization wait time                               │
│  - Load imbalance (unequal work distribution)              │
│  - Cache coherence traffic                                 │
│  - Memory contention                                        │
│                                                             │
│  Efficiency graph:                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Efficiency                                           │   │
│  │ 100%│────┐                                          │   │
│  │     │    └────┐                                     │   │
│  │ 80% │         └────┐                                │   │
│  │     │              └────┐                           │   │
│  │ 60% │                   └────┐                      │   │
│  │     │                        └────┐                 │   │
│  │ 40% │                             └────┐            │   │
│  │     │                                  └────        │   │
│  │ 20% │                                              │   │
│  │     └─────────────────────────────────────────     │   │
│  │     1   2   4   8   16  32  64  128 Number of processors│
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Generally efficiency decreases as processor count increases│
│  (Diminishing returns)                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Synchronization and Locks

### 7.1 Need for Synchronization

```
Race Condition example:

┌─────────────────────────────────────────────────────────────┐
│  Shared variable: counter = 0                               │
│                                                             │
│  Thread 0              Thread 1                             │
│  ─────────────────     ─────────────────                    │
│  load  counter         load  counter         // Both 0     │
│  add   1               add   1               // Both 1     │
│  store counter         store counter         // Both store 1│
│                                                             │
│  Expected result: counter = 2                               │
│  Actual result: counter = 1  ← One increment lost!         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Atomic Operations

```
Hardware-supported atomic operations:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Test-and-Set (TAS):                                        │
│  // Why atomicity matters: if read and write were separate   │
│  // instructions, two threads could both read 0 ("unlocked") │
│  // before either writes 1 — both would believe they acquired│
│  // the lock. The hardware LOCK prefix on x86 ensures the    │
│  // read-modify-write completes as one indivisible operation. │
│  ┌───────────────────────────────────────────────────┐     │
│  │  int TAS(int *lock) {                             │     │
│  │      int old = *lock;   // Read                   │     │
│  │      *lock = 1;         // Write      Executed    │     │
│  │      return old;        // Return     atomically  │     │
│  │  }                                                │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  Compare-and-Swap (CAS):                                    │
│  ┌───────────────────────────────────────────────────┐     │
│  │  bool CAS(int *addr, int expected, int new) {     │     │
│  │      if (*addr == expected) {                     │     │
│  │          *addr = new;                Executed     │     │
│  │          return true;                atomically   │     │
│  │      }                                            │     │
│  │      return false;                                │     │
│  │  }                                                │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  Fetch-and-Add:                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │  int FAA(int *addr, int val) {                    │     │
│  │      int old = *addr;                 Executed    │     │
│  │      *addr = old + val;               atomically  │     │
│  │      return old;                                  │     │
│  │  }                                                │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  x86 instruction examples:                                  │
│  - LOCK XCHG (Test-and-Set)                                │
│  - LOCK CMPXCHG (Compare-and-Swap)                         │
│  - LOCK XADD (Fetch-and-Add)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Spinlock

```
┌─────────────────────────────────────────────────────────────┐
│                    Spinlock Implementation                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Simple spinlock (Test-and-Set):                            │
│  ┌───────────────────────────────────────────────────┐     │
│  │  void lock(int *lock) {                           │     │
│  │      while (TAS(lock) == 1) {                     │     │
│  │          // Spin (busy wait)                      │     │
│  │      }                                            │     │
│  │  }                                                │     │
│  │                                                   │     │
│  │  void unlock(int *lock) {                         │     │
│  │      *lock = 0;                                   │     │
│  │  }                                                │     │
│  └───────────────────────────────────────────────────┘     │
│  Problem: Bus traffic on every TAS                          │
│                                                             │
│  Test-and-Test-and-Set (TTAS):                              │
│  // Why TTAS is much better than plain TAS: each TAS issues │
│  // a LOCK'd bus transaction that forces all other caches to │
│  // invalidate their copy — even when the lock is held and   │
│  // the attempt will fail. With TTAS, the inner while loop   │
│  // reads from the local cache (Shared state) generating zero│
│  // bus traffic. Only when the value changes (lock released) │
│  // does it attempt the expensive TAS. This reduces bus      │
│  // contention from O(N) per spin iteration to O(1) on       │
│  // release, where N is the number of waiting cores.         │
│  ┌───────────────────────────────────────────────────┐     │
│  │  void lock(int *lock) {                           │     │
│  │      while (1) {                                  │     │
│  │          while (*lock == 1) {                     │     │
│  │              // Spin in local cache (no bus traffic)│   │
│  │          }                                        │     │
│  │          if (TAS(lock) == 0) {                    │     │
│  │              return;  // Lock acquired            │     │
│  │          }                                        │     │
│  │      }                                            │     │
│  │  }                                                │     │
│  └───────────────────────────────────────────────────┘     │
│  Improvement: Wait in local cache until lock released       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.4 Mutex and Semaphore

```
┌─────────────────────────────────────────────────────────────┐
│                    Synchronization Primitives                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Mutex:                                                     │
│  - Mutual Exclusion                                        │
│  - Only one thread allowed at a time                       │
│  - Has ownership (only acquiring thread can release)       │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │  pthread_mutex_t mutex;                           │     │
│  │  pthread_mutex_init(&mutex, NULL);                │     │
│  │                                                   │     │
│  │  pthread_mutex_lock(&mutex);                      │     │
│  │  // Critical Section                              │     │
│  │  pthread_mutex_unlock(&mutex);                    │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  Semaphore:                                                 │
│  - Counter-based synchronization                            │
│  - Can allow N threads simultaneous access                  │
│  - No ownership                                             │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │  sem_t sem;                                       │     │
│  │  sem_init(&sem, 0, 3);  // Max 3 concurrent access│     │
│  │                                                   │     │
│  │  sem_wait(&sem);    // Decrement counter, wait if 0│    │
│  │  // Critical Section                              │     │
│  │  sem_post(&sem);    // Increment counter, wake waiters│  │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.5 Lock-Free Algorithms

```
┌─────────────────────────────────────────────────────────────┐
│               Lock-Free Counter Example                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  // Why the CAS retry loop works: if no other thread modified│
│  // counter between our load and CAS, the exchange succeeds │
│  // and we're done. If another thread did increment it, CAS │
│  // fails, 'old' is updated to the current value, and we    │
│  // retry with the fresh value. Progress is guaranteed       │
│  // because at least one thread succeeds per round.          │
│  void increment(atomic_int *counter) {                      │
│      int old, new;                                         │
│      do {                                                  │
│          old = atomic_load(counter);                       │
│          new = old + 1;                                    │
│      } while (!atomic_compare_exchange(counter, &old, new));│
│  }                                                         │
│                                                             │
│  Operation:                                                 │
│  1. Read current value                                     │
│  2. Calculate new value                                    │
│  3. Attempt atomic update with CAS                         │
│  4. Retry if failed (another thread modified it)           │
│                                                             │
│  Advantages:                                                │
│  - No lock waiting                                         │
│  - Deadlock impossible                                     │
│  - No priority inversion                                   │
│                                                             │
│  Disadvantages:                                             │
│  - Complex implementation                                  │
│  - Must be careful of ABA problem                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. GPU and Parallel Computing

### 8.1 GPU Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU vs CPU Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CPU (Latency Optimized):                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │              Large Cache                     │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │  ┌─────────────┐ ┌─────────────┐                    │   │
│  │  │   Complex   │ │   Complex   │                    │   │
│  │  │   Core 0    │ │   Core 1    │  (4-16 cores)      │   │
│  │  │  (OoO, BP)  │ │  (OoO, BP)  │                    │   │
│  │  └─────────────┘ └─────────────┘                    │   │
│  └─────────────────────────────────────────────────────┘   │
│  Features: Complex cores, large cache, low latency         │
│                                                             │
│  GPU (Throughput Optimized):                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│   │
│  │  │ S │ S │ S │ S │ S │ S │ S │ S │ S │ S │ S │ S ││   │
│  │  │ M │ M │ M │ M │ M │ M │ M │ M │ M │ M │ M │ M ││   │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘│   │
│  │     (Thousands of simple cores/CUDA Cores)          │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │           Small Cache per SM                │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│  Features: Thousands of simple cores, small cache, high throughput│
│                                                             │
│  SM (Streaming Multiprocessor) structure:                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  32 CUDA Cores (1 Warp = 32 threads)        │    │   │
│  │  │  Each core executes same instruction (SIMT)  │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │  + Shared Memory (48KB)                              │   │
│  │  + Register File (65536 × 32bit)                    │   │
│  │  + Warp Scheduler                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 CUDA Programming Model

```
┌─────────────────────────────────────────────────────────────┐
│                    CUDA Execution Model                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Hierarchy:                                                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                      Grid                           │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │   │
│  │  │ Block   │ │ Block   │ │ Block   │ │ Block   │  │   │
│  │  │ (0,0)   │ │ (1,0)   │ │ (2,0)   │ │ (3,0)   │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘  │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │   │
│  │  │ Block   │ │ Block   │ │ Block   │ │ Block   │  │   │
│  │  │ (0,1)   │ │ (1,1)   │ │ (2,1)   │ │ (3,1)   │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Inside Block:                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐│   │
│  │  │ T0  │ T1  │ T2  │ T3  │ T4  │ T5  │ ... │T255 ││   │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘│   │
│  │                                                     │   │
│  │  256 Threads sharing Shared Memory                  │   │
│  │  Synchronization: __syncthreads()                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Memory hierarchy:                                          │
│  - Global Memory: All threads access, slow (~500 cycles)   │
│  - Shared Memory: Shared within block, fast (~5 cycles)    │
│  - Register: Thread-private, fastest                       │
│  - Constant Memory: Read-only, cached                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 CUDA Code Example

```cuda
// Vector addition CUDA kernel
// Why __global__: this qualifier tells the NVCC compiler that vectorAdd runs on
// the GPU (device) but is callable from the CPU (host). Each of the million
// threads executes this same function body but with a different idx value —
// this is the SIMT (Single Instruction, Multiple Threads) execution model.

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    // Calculate global thread index
    // Why this formula: blockIdx.x identifies which block this thread belongs to,
    // blockDim.x is the number of threads per block (256), and threadIdx.x is the
    // thread's position within its block. Together they produce a unique index
    // from 0 to N-1 across all threads in the grid.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Why the bounds check: the grid may launch more threads than N elements
    // (since blocksPerGrid rounds up). Without this guard, out-of-bounds threads
    // would read/write past the end of the arrays — causing memory corruption.
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data (Host → Device)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel execution configuration
    // Why 256 threads per block: this is a common sweet spot. GPU SMs have a
    // limited number of registers and shared memory; 256 threads balances
    // occupancy (enough warps to hide memory latency) against resource pressure.
    // Too few threads → SM idles; too many → register spilling to slow local memory.
    int threadsPerBlock = 256;
    // Why round up: if N=1000000 and threadsPerBlock=256, we need ceil(1000000/256)
    // = 3907 blocks. The (N + 255) / 256 formula is the integer ceiling trick.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Execute kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result (Device → Host)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
```

### 8.4 GPU vs CPU Use Cases

```
┌─────────────────────────────────────────────────────────────┐
│                GPU vs CPU Suitable Tasks                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GPU suitable (Data parallel):                             │
│  - Matrix operations                                       │
│  - Image/video processing                                  │
│  - Deep learning training/inference                        │
│  - Physics simulation                                      │
│  - Cryptocurrency mining                                   │
│  - Scientific computing (CFD, molecular dynamics)          │
│                                                             │
│  CPU suitable (Task parallel, complex control flow):       │
│  - Operating systems                                       │
│  - Databases                                               │
│  - Web servers                                             │
│  - Compilers                                               │
│  - General applications                                    │
│                                                             │
│  Performance comparison (approximate):                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Task               │  CPU     │  GPU      │ Ratio  │    │
│  ├────────────────────────────────────────────────────┤    │
│  │ Matrix mult (4K×4K)│  10s     │  0.1s    │ 100x  │    │
│  │ Image filter       │  2s      │  0.05s   │ 40x   │    │
│  │ Neural net training│  100s    │  2s      │ 50x   │    │
│  │ Sorting algorithm  │  5s      │  4s      │ 1.25x │    │
│  │ Branch-heavy code  │  1s      │  10s     │ 0.1x  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Practice Problems

### Basic Problems

1. Explain the four classifications of Flynn's Taxonomy.

2. What is the difference between SMP and NUMA?

3. Explain the four states of the MESI protocol.

### Intermediate Problems

4. When 80% of a program is parallelizable, what is the maximum performance improvement on an 8-core system according to Amdahl's Law?

5. Explain the Race Condition that can occur in the following code and provide a solution:
   ```c
   int counter = 0;
   void increment() {
       counter++;
   }
   ```

6. Explain the difference between Test-and-Set and Compare-and-Swap.

### Advanced Problems

7. Track the state transitions in the MESI protocol for the following scenario:
   - Core 0 reads X (from memory)
   - Core 1 reads X
   - Core 0 writes X
   - Core 1 reads X

8. Explain why GPUs are faster than CPUs for matrix multiplication.

9. Explain the advantages and disadvantages of Lock-Free algorithms and the ABA problem.

<details>
<summary>Answers</summary>

1. Flynn's Taxonomy:
   - SISD: Single Instruction Single Data (traditional CPU)
   - SIMD: Single Instruction Multiple Data (vector operations, GPU)
   - MISD: Multiple Instruction Single Data (rarely used)
   - MIMD: Multiple Instruction Multiple Data (multicore)

2. SMP vs NUMA:
   - SMP: All CPUs have uniform memory access (UMA)
   - NUMA: Local memory access faster than remote, better scalability

3. MESI states:
   - Modified: Modified, sole copy
   - Exclusive: Not modified, sole copy
   - Shared: Not modified, multiple copies possible
   - Invalid: Invalid

4. Amdahl's Law calculation:
   Speedup = 1 / (0.2 + 0.8/8) = 1 / (0.2 + 0.1) = 1/0.3 = 3.33x

5. Race Condition solution:
   - Problem: counter++ is not atomic (read-modify-write)
   - Solution: Use mutex, or use atomic operation (atomic_fetch_add)

6. TAS vs CAS:
   - TAS: Always sets to 1, returns previous value
   - CAS: Sets to new value only if matches expected value

7. MESI state transitions:
   - Core 0 read: Core0=E, Core1=I
   - Core 1 read: Core0=S, Core1=S
   - Core 0 write: Core0=M, Core1=I (invalidated)
   - Core 1 read: Core0=S, Core1=S (Core0 provides data)

8. Why GPUs are faster for matrix multiplication:
   - High data parallelism (each element computed independently)
   - Thousands of cores executing simultaneously
   - High memory bandwidth
   - Matrix multiplication optimized for GPU architecture

9. Lock-Free algorithms:
   - Advantages: No lock waiting, deadlock impossible
   - Disadvantages: Complex implementation, difficult debugging
   - ABA problem: Value changes A→B→A, CAS still succeeds
   - Solution: Use tag/version counter, Hazard Pointers

</details>

---

## References

- Computer Architecture: A Quantitative Approach (Hennessy & Patterson)
- The Art of Multiprocessor Programming (Herlihy & Shavit)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Intel Optimization Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Memory Consistency Models](https://research.swtch.com/hwmm)
