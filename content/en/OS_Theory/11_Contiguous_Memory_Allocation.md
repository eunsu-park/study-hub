# Contiguous Memory Allocation ⭐⭐

**Previous**: [Memory Management Basics](./10_Memory_Management_Basics.md) | **Next**: [Paging](./12_Paging.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain contiguous memory allocation and its limitations
2. Compare first-fit, best-fit, and worst-fit allocation strategies
3. Distinguish internal from external fragmentation
4. Explain compaction and when it is feasible
5. Analyze why contiguous allocation motivates the move to paging

---

The simplest approach to memory allocation -- give each process one contiguous block -- seems intuitive but leads to a fundamental problem: over time, free memory becomes scattered into small unusable fragments. Understanding this fragmentation problem explains why modern systems use paging instead.

## Table of Contents

1. [Memory Partitioning Overview](#1-memory-partitioning-overview)
2. [Fixed Partitioning](#2-fixed-partitioning)
3. [Variable Partitioning](#3-variable-partitioning)
4. [Memory Placement Strategies](#4-memory-placement-strategies)
5. [Fragmentation](#5-fragmentation)
6. [Compaction](#6-compaction)
7. [Practice Problems](#7-practice-problems)

---

## 1. Memory Partitioning Overview

### Memory Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Overall Memory Structure                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Low Address                                                │
│   ┌──────────────────────────────────────────┐              │
│   │      Operating System (Kernel)            │  0x0000      │
│   │ Interrupt vectors, drivers, kernel code   │              │
│   ├──────────────────────────────────────────┤              │
│   │                                          │              │
│   │                                          │              │
│   │          User Area                       │              │
│   │      (Used by processes)                 │              │
│   │                                          │              │
│   │                                          │              │
│   └──────────────────────────────────────────┘  0xFFFF      │
│   High Address                                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Fixed Partitioning

### 2.1 Concept

Divide memory into fixed-size partitions in advance.

```
┌─────────────────────────────────────────────────────────────┐
│                  Fixed Partitioning (Equal Size)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │                  OS                       │ 64KB         │
│  ├──────────────────────────────────────────┤               │
│  │              Partition 1                  │ 64KB         │
│  ├──────────────────────────────────────────┤               │
│  │              Partition 2                  │ 64KB         │
│  ├──────────────────────────────────────────┤               │
│  │              Partition 3                  │ 64KB         │
│  ├──────────────────────────────────────────┤               │
│  │              Partition 4                  │ 64KB         │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  Feature: All partitions same size (64KB)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Fixed Partitioning Problems

```
┌─────────────────────────────────────────────────────────────┐
│                 Internal Fragmentation Occurs                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Allocating 45KB process to 64KB partition:                 │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │          Process (45KB)                   │ Used         │
│  ├──────────────────────────────────────────┤               │
│  │          Waste (19KB)                    │ Internal Frag │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  → 19KB wasted (cannot be used by other processes)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Variable Partitioning

### 3.1 Concept

Dynamically create partitions to match process sizes.

```
┌─────────────────────────────────────────────────────────────┐
│                    Variable Partitioning Example             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Initial state:                                              │
│  ┌──────────────────────────────────────────┐ 0             │
│  │                  OS                       │ 64KB         │
│  ├──────────────────────────────────────────┤ 64KB         │
│  │                                          │               │
│  │             Free Space                    │               │
│  │              (448KB)                     │               │
│  │                                          │               │
│  └──────────────────────────────────────────┘ 512KB        │
│                                                              │
│  After P1(100KB), P2(50KB), P3(200KB) loaded:               │
│  ┌──────────────────────────────────────────┐ 0             │
│  │                  OS                       │ 64KB         │
│  ├──────────────────────────────────────────┤               │
│  │             P1 (100KB)                   │               │
│  ├──────────────────────────────────────────┤ 164KB        │
│  │          P2 (50KB)                       │               │
│  ├──────────────────────────────────────────┤ 214KB        │
│  │                                          │               │
│  │             P3 (200KB)                   │               │
│  │                                          │               │
│  ├──────────────────────────────────────────┤ 414KB        │
│  │         Free Space (98KB)                │               │
│  └──────────────────────────────────────────┘ 512KB        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Memory Placement Strategies

Strategies for deciding which hole to place a new process in.

### 4.1 First-Fit

```
┌─────────────────────────────────────────────────────────────┐
│                    First-Fit Strategy                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Free space list:                                            │
│  [100KB] -> [500KB] -> [200KB] -> [300KB]                   │
│                                                              │
│  150KB process allocation request:                           │
│                                                              │
│  1. Check [100KB] → 150KB > 100KB → Cannot                  │
│  2. Check [500KB] → 150KB <= 500KB → Allocate here!         │
│                                                              │
│  Result:                                                     │
│  [100KB] -> [350KB] -> [200KB] -> [300KB]                   │
│              ↑ (500-150=350KB remaining)                     │
│                                                              │
│  Advantage: Fast search                                      │
│  Disadvantage: Small holes accumulate at front               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Best-Fit

```
┌─────────────────────────────────────────────────────────────┐
│                    Best-Fit Strategy                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Free space list:                                            │
│  [100KB] -> [500KB] -> [200KB] -> [300KB]                   │
│                                                              │
│  150KB process allocation request:                           │
│                                                              │
│  Full search:                                                │
│  - [100KB]: Cannot (100 < 150)                               │
│  - [500KB]: Possible, remaining = 350KB                      │
│  - [200KB]: Possible, remaining = 50KB  ← Minimum waste!    │
│  - [300KB]: Possible, remaining = 150KB                      │
│                                                              │
│  Allocate to 200KB block!                                    │
│                                                              │
│  Result:                                                     │
│  [100KB] -> [500KB] -> [50KB] -> [300KB]                    │
│                                                              │
│  Advantage: Minimize memory waste                            │
│  Disadvantage: Full search needed, creates tiny holes        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Worst-Fit

```
┌─────────────────────────────────────────────────────────────┐
│                   Worst-Fit Strategy                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Free space list:                                            │
│  [100KB] -> [500KB] -> [200KB] -> [300KB]                   │
│                                                              │
│  150KB process allocation request:                           │
│                                                              │
│  Find largest block:                                         │
│  - [100KB], [500KB], [200KB], [300KB]                       │
│  - Maximum: 500KB ← Allocate here!                          │
│                                                              │
│  Result:                                                     │
│  [100KB] -> [350KB] -> [200KB] -> [300KB]                   │
│              ↑ (Large remaining space = usable later)       │
│                                                              │
│  Advantage: Large remaining space can fit other processes    │
│  Disadvantage: Difficulty placing large processes            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Fragmentation

### 5.1 Internal Fragmentation

```
┌─────────────────────────────────────────────────────────────┐
│                     Internal Fragmentation                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Occurs in fixed partitioning or paging:                     │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │                                          │               │
│  │   Allocated block (e.g., 4KB page)       │               │
│  │                                          │               │
│  │  ┌────────────────────────┐              │               │
│  │  │ Process data (3KB)     │              │               │
│  │  ├────────────────────────┤              │               │
│  │  │ Internal frag (1KB)    │ ← Waste!     │               │
│  │  └────────────────────────┘              │               │
│  │                                          │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  Features:                                                   │
│  - Waste inside allocated block                             │
│  - Cannot be used by other processes                         │
│  - Occurs in fixed partitioning, paging                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 External Fragmentation

```
┌─────────────────────────────────────────────────────────────┐
│                     External Fragmentation                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Occurs in variable partitioning:                            │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │  OS                                      │               │
│  ├──────────────────────────────────────────┤               │
│  │  P1 (100KB)                              │               │
│  ├──────────────────────────────────────────┤               │
│  │  *** Hole (30KB) ***                     │ ← Small hole  │
│  ├──────────────────────────────────────────┤               │
│  │  P2 (200KB)                              │               │
│  ├──────────────────────────────────────────┤               │
│  │  *** Hole (25KB) ***                     │ ← Small hole  │
│  ├──────────────────────────────────────────┤               │
│  │  P3 (150KB)                              │               │
│  ├──────────────────────────────────────────┤               │
│  │  *** Hole (45KB) ***                     │ ← Small hole  │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  Total free space: 30 + 25 + 45 = 100KB                     │
│  But cannot load 50KB process! (No contiguous space)         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Compaction

### 6.1 Concept

Move all processes to one end to create large contiguous free space, solving external fragmentation.

```
┌─────────────────────────────────────────────────────────────┐
│                      Compaction Process                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Before compaction:                After compaction:         │
│                                                              │
│  ┌──────────────┐                 ┌──────────────┐          │
│  │  OS          │                 │  OS          │          │
│  ├──────────────┤                 ├──────────────┤          │
│  │  P1 (100KB)  │                 │  P1 (100KB)  │          │
│  ├──────────────┤                 ├──────────────┤          │
│  │  Hole (30KB) │    ────▶        │  P2 (200KB)  │          │
│  ├──────────────┤                 │              │          │
│  │  P2 (200KB)  │                 ├──────────────┤          │
│  │              │                 │  P3 (150KB)  │          │
│  ├──────────────┤                 │              │          │
│  │  Hole (25KB) │                 ├──────────────┤          │
│  ├──────────────┤                 │              │          │
│  │  P3 (150KB)  │                 │  Hole (100KB)│          │
│  │              │                 │  (contiguous)│          │
│  ├──────────────┤                 │              │          │
│  │  Hole (45KB) │                 │              │          │
│  └──────────────┘                 └──────────────┘          │
│                                                              │
│  Total hole: 100KB                Total hole: 100KB (contig)│
│  (Scattered)                      New process can load!     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Problem 1: Placement Strategy
Given free blocks: [200KB, 80KB, 300KB, 150KB]

For a 120KB process, which block is selected by each strategy?
1. First-Fit
2. Best-Fit
3. Worst-Fit

<details>
<summary>Show Answer</summary>

1. First-Fit: 200KB (first block that fits)
2. Best-Fit: 150KB (smallest waste: 150-120=30KB)
3. Worst-Fit: 300KB (largest block)

</details>

---

## Exercises

### Exercise 1: Placement Strategy Simulation

A memory system has 512KB total. The OS uses 64KB. After several allocations and deallocations, the current free list (in address order) is:

```
[Hole at 64KB: 80KB] [Hole at 256KB: 120KB] [Hole at 440KB: 72KB]
```

Three processes arrive: P1=90KB, P2=70KB, P3=115KB (in order).

For each of **First-Fit**, **Best-Fit**, and **Worst-Fit**:
1. Show which hole is selected for each process (in arrival order)
2. Draw the memory state after all three allocations (or indicate failure)
3. Calculate total external fragmentation remaining after all allocations

### Exercise 2: Fragmentation Analysis

A system uses **fixed partitioning** with four equal partitions of 256KB each (total 1MB). Over time, the following processes are loaded:

| Process | Size | Partition Assigned |
|---------|------|-------------------|
| P1 | 200KB | Partition 1 |
| P2 | 50KB | Partition 2 |
| P3 | 255KB | Partition 3 |
| P4 | 100KB | Partition 4 |

1. Calculate the internal fragmentation in each partition
2. Calculate the total internal fragmentation
3. P5 (270KB) arrives. Can it be loaded? Why not? What type of fragmentation is this?
4. If the same four processes used **variable partitioning** instead, what would the internal fragmentation be? What is the total free space?

### Exercise 3: Compaction Cost-Benefit

A system with 1MB memory has reached this state (after multiple allocations and deallocations):

```
[OS: 100KB] [P1: 150KB] [Hole: 50KB] [P2: 200KB] [Hole: 80KB] [P3: 100KB] [Hole: 320KB]
```

P4 (350KB) wants to load but cannot fit in any single hole.

1. What is the total free memory? Why can't P4 load despite having enough total free space?
2. Compaction requires copying all in-memory processes to close the holes. How much data must be moved (in KB)?
3. If memory bandwidth is 10 GB/s, how long does compaction take? Is this acceptable for an interactive system?
4. After compaction, draw the new memory layout and confirm P4 can load
5. What hardware requirement is necessary for compaction to work safely (i.e., so existing processes don't crash after being moved)?

### Exercise 4: Variable Partitioning Trace

Starting from an empty 1MB (1024KB) memory where the OS occupies the first 128KB, apply **First-Fit** allocation for the following sequence of events:

| Event | Action |
|-------|--------|
| t=1 | P1 (200KB) arrives |
| t=2 | P2 (300KB) arrives |
| t=3 | P3 (150KB) arrives |
| t=4 | P1 terminates |
| t=5 | P4 (100KB) arrives |
| t=6 | P3 terminates |
| t=7 | P5 (400KB) arrives |

For each event:
1. Draw the memory layout showing allocated regions and holes
2. Note the address at which each process is placed
3. At t=7, does P5 load successfully? If not, would Best-Fit or Worst-Fit help? Would compaction help?

### Exercise 5: Strategy Evaluation

You are designing a memory manager for a real-time embedded system with 256KB RAM and the following characteristics:
- 8 fixed-size tasks (each exactly 16KB) that load and unload frequently
- 2 variable-size tasks (between 20KB and 60KB) that run occasionally
- Allocation must complete in under 10 microseconds to meet real-time deadlines
- Memory cannot be compacted (copying data would violate real-time constraints)

1. Is fixed partitioning or variable partitioning more appropriate for the 8 fixed-size tasks? Calculate the internal fragmentation with your choice.
2. For the 2 variable-size tasks, what placement strategy minimizes fragmentation while keeping allocation time predictable?
3. Over time, will external fragmentation become a problem with your design? Why or why not?
4. What data structure would you use to track the free list, and why? Consider both search time and update time when tasks arrive and depart.

---

## Next Steps

Next topic: Paging

---

## References

- Silberschatz, "Operating System Concepts" Chapter 8
- Tanenbaum, "Modern Operating Systems" Chapter 3
