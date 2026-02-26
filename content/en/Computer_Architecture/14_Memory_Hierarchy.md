# Memory Hierarchy

**Previous**: [13_Superscalar_Out_of_Order.md](./13_Superscalar_Out_of_Order.md) | **Next**: [15_Cache_Memory.md](./15_Cache_Memory.md)

**Difficulty**: ⭐⭐

**Prerequisites**: Computer System Overview, CPU Architecture Basics

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the memory hierarchy from registers to disk and explain the speed-cost tradeoff at each level
2. Explain temporal and spatial locality and why programs exhibit them
3. Calculate effective memory access time given hit rates and access times
4. Explain why the memory wall problem exists (CPU speed vs memory speed gap)
5. Describe how the memory hierarchy creates the illusion of large, fast memory

---

CPU speeds have increased exponentially, but memory speeds have not kept pace -- this is the "memory wall." The memory hierarchy is the computer architect's elegant solution: a cascade of increasingly larger but slower memories, designed so that the data the CPU needs most is almost always in the fastest layer. Understanding the hierarchy explains why data structures with good locality (arrays) outperform those with poor locality (linked lists) by orders of magnitude.

## Table of Contents

1. [The Need for Memory Hierarchy](#1-the-need-for-memory-hierarchy)
2. [Principle of Locality](#2-principle-of-locality)
3. [Memory Technology Comparison](#3-memory-technology-comparison)
4. [Memory Levels](#4-memory-levels)
5. [Memory Bandwidth and Latency](#5-memory-bandwidth-and-latency)
6. [Memory Performance Optimization](#6-memory-performance-optimization)
7. [Practice Problems](#7-practice-problems)

---

## 1. The Need for Memory Hierarchy

### 1.1 CPU-Memory Speed Gap

```
CPU and Memory Performance Trends (1980-2020):

Performance
  │
  │                                        ●────────── CPU
  │                                   ●
  │                              ●
  │                         ●
  │                    ●
  │               ●
  │          ●                               ◆───────── Memory
  │     ●                              ◆
  │ ●                            ◆
  │                        ◆
  │                  ◆
  │            ◆
  │      ◆
  │ ◆
  └─────────────────────────────────────────────────── Year
   1980    1990    2000    2010    2020

CPU: ~50% performance improvement per year (1980-2000)
Memory: ~7% performance improvement per year

Result: "Memory Wall" - Memory becomes the system performance bottleneck
```

### 1.2 Ideal Memory vs Reality

```
Ideal Memory:
┌─────────────────────────────────────────┐
│  - Infinite capacity                     │
│  - Zero access time                      │
│  - Free                                  │
└─────────────────────────────────────────┘

Reality:
┌─────────────────────────────────────────────────────────────┐
│  Speed ↑ = Cost ↑, Capacity ↓                               │
│  Capacity ↑ = Cost ↑, Speed ↓                               │
│  Cost ↓ = Speed ↓, Limited capacity                         │
└─────────────────────────────────────────────────────────────┘

Solution: Memory Hierarchy
- Arrange multiple types of memory hierarchically
- Fast and small memory close to CPU
- Slow and large memory farther away
- Keep frequently used data in fast memory
```

### 1.3 Memory Hierarchy Concept

```
                        ┌───────────┐
                        │ Registers │  ← Fastest, smallest, most expensive
                        │   ~1KB    │
                        └─────┬─────┘
                              │
                        ┌─────┴─────┐
                        │ L1 Cache  │
                        │  ~64KB    │
                        └─────┬─────┘
                              │
                    ┌─────────┴─────────┐
                    │     L2 Cache      │
                    │    ~256KB-1MB     │
                    └─────────┬─────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │            L3 Cache               │
            │           ~2-32MB                 │
            └─────────────────┬─────────────────┘
                              │
    ┌─────────────────────────┴─────────────────────────┐
    │                   Main Memory                      │
    │                  ~8-128GB                          │
    └─────────────────────────┬─────────────────────────┘
                              │
┌─────────────────────────────┴─────────────────────────────┐
│                        SSD/HDD                             │
│                     ~256GB-10TB                            │  ← Slowest, largest, cheapest
└───────────────────────────────────────────────────────────┘

Going up: Faster, smaller, more expensive, closer to CPU
Going down: Slower, larger, cheaper, farther from CPU
```

---

## 2. Principle of Locality

### 2.1 What is Locality?

There are patterns in how programs access memory. This is called **locality**, and it is why memory hierarchy is effective.

### 2.2 Temporal Locality

```
Definition: Data accessed recently is likely to be accessed again in the near future

Example 1: Loop variables
┌─────────────────────────────────────┐
│ for (i = 0; i < 1000; i++) {       │
│     sum += array[i];               │
│ }                                  │
└─────────────────────────────────────┘
  Variables 'sum' and 'i' are accessed 1000 times

Example 2: Frequently called functions
┌─────────────────────────────────────┐
│ while (running) {                  │
│     process_event();  // repeated  │
│     update_state();   // repeated  │
│ }                                  │
└─────────────────────────────────────┘
  Function code is reused continuously

Exploiting temporal locality:
- Keep recently accessed data in cache
- Provide quickly from cache on re-access
```

### 2.3 Spatial Locality

```
Definition: Data near accessed data is likely to be accessed soon

Example 1: Sequential array access
┌─────────────────────────────────────┐
│ for (i = 0; i < N; i++) {          │
│     sum += array[i];               │
│ }                                  │
└─────────────────────────────────────┘

Memory layout:
Address:  0x100  0x104  0x108  0x10C  0x110  0x114
          ┌──────┬──────┬──────┬──────┬──────┬──────┐
array:    │ a[0] │ a[1] │ a[2] │ a[3] │ a[4] │ a[5] │
          └──────┴──────┴──────┴──────┴──────┴──────┘
            ↓      ↓      ↓      ↓      ↓      ↓
          Sequential access - high spatial locality

Example 2: Struct access
┌─────────────────────────────────────┐
│ struct Point { int x, y, z; };     │
│ Point p;                           │
│ distance = sqrt(p.x*p.x +          │
│                 p.y*p.y +          │
│                 p.z*p.z);          │
└─────────────────────────────────────┘
  p.x, p.y, p.z are adjacent in memory

Exploiting spatial locality:
- Load data in cache line (block) units
- Fetch multiple adjacent data at once
- 64-byte cache line = 16 ints included
```

### 2.4 Optimization Using Locality

```
Good code (high locality):

// Row-major traversal - matches memory layout
// Why row-major is fast: C stores matrix[i][j] so that consecutive j values
// are contiguous in memory. A 64-byte cache line holds 16 ints, so accessing
// j, j+1, j+2, ... gets 15 hits per 1 miss (spatial locality).
for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
        sum += matrix[i][j];
    }
}

Memory access pattern:
[0,0] [0,1] [0,2] [0,3] ... [1,0] [1,1] ...
  ↓     ↓     ↓     ↓        ↓     ↓
Sequential access → High cache hit rate


Bad code (low locality):

// Column-major traversal - mismatches memory layout
// Why column-major is slow: each increment of i jumps N elements (N×4 bytes) in
// memory, landing in a completely different cache line every time. For N=1024 with
// 4-byte ints, successive accesses are 4096 bytes apart — 64× the cache line size.
// Every single access is a cache miss, destroying spatial locality.
for (j = 0; j < N; j++) {
    for (i = 0; i < N; i++) {
        sum += matrix[i][j];
    }
}

Memory access pattern:
[0,0] [1,0] [2,0] [3,0] ... [0,1] [1,1] ...
  ↓     ↓     ↓     ↓        ↓     ↓
Jumping N elements → Frequent cache misses


Performance difference (N=1024, 8-byte elements):
- Row-major: ~50ms
- Column-major: ~500ms (10x slower)
```

### 2.5 Working Set

```
Definition: The memory region actively used by a program during a specific time interval

┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Working Set changes by program execution phase              │
│                                                              │
│  Phase 1: Initialization                                     │
│  Working Set: [init code] + [config data]                   │
│  Size: ~100KB                                                │
│                                                              │
│  Phase 2: Data Processing                                    │
│  Working Set: [processing code] + [input buffer] + [output] │
│  Size: ~10MB                                                 │
│                                                              │
│  Phase 3: Result Output                                      │
│  Working Set: [output code] + [output buffer]               │
│  Size: ~500KB                                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Cache size and Working Set:
- Working Set < Cache size: High hit rate
- Working Set > Cache size: Thrashing may occur
```

---

## 3. Memory Technology Comparison

### 3.1 SRAM (Static RAM)

```
SRAM Cell Structure (6T SRAM):

        Vdd                         Vdd
         │                           │
      ┌──┴──┐                     ┌──┴──┐
      │ P1  │                     │ P2  │
      └──┬──┘                     └──┬──┘
         │          ┌───┐            │
         ├──────────┤   ├────────────┤
         │          └───┘            │
      ┌──┴──┐                     ┌──┴──┐
      │ N1  │        Word         │ N2  │
      └──┬──┘        Line         └──┬──┘
         │            │              │
         │         ┌──┴──┐           │
         │         │ N3  │           │
         │         └──┬──┘           │
         │            │              │
        Bit         GND            ~Bit
        Line                       Line

Characteristics:
- 6 transistors store 1 bit
- Data retained as long as power is supplied (no refresh needed)
- Fast access speed (~1-2ns)
- High cost, low density
- Usage: Cache memory, register files
```

### 3.2 DRAM (Dynamic RAM)

```
DRAM Cell Structure (1T1C DRAM):

    Word Line
        │
     ┌──┴──┐
     │  T  │ (transistor)
     └──┬──┘
        │
     ┌──┴──┐
     │  C  │ (capacitor)
     └──┬──┘
        │
    Bit Line

Characteristics:
- 1 transistor + 1 capacitor stores 1 bit
- Periodic refresh needed due to capacitor leakage (~64ms)
- Slow access speed (~50-100ns)
- Low cost, high density
- Usage: Main memory (DDR4, DDR5)

DRAM Access Process:
1. Row Activate (RAS): Select row, move data to sense amplifier
2. Column Read (CAS): Select column, output data
3. Precharge: Prepare for next access

┌─────────────────────────────────────────────────────┐
│ RAS Latency │ CAS Latency │ Precharge │ = Total delay │
└─────────────────────────────────────────────────────┘
    ~15ns        ~15ns         ~15ns       ~45ns+
```

### 3.3 Flash Memory (SSD)

```
NAND Flash Structure:

     Word Line 0  Word Line 1  Word Line 2
          │           │           │
    ┌─────┼───────────┼───────────┼─────┐
    │  ┌──┴──┐     ┌──┴──┐     ┌──┴──┐  │
    │  │Cell │     │Cell │     │Cell │  │ String 0
    │  └──┬──┘     └──┬──┘     └──┬──┘  │
    │     │           │           │     │
    │  ┌──┴──┐     ┌──┴──┐     ┌──┴──┐  │
    │  │Cell │     │Cell │     │Cell │  │ String 1
    │  └──┬──┘     └──┬──┘     └──┬──┘  │
    └─────┼───────────┼───────────┼─────┘
          │           │           │
       Bit Line    Bit Line    Bit Line

Characteristics:
- Stores charge in floating gate
- Non-volatile (data retained without power)
- Read: ~25us, Write: ~250us, Erase: ~2ms
- Only block-level erase possible
- Usage: SSD, USB drives, SD cards
```

### 3.4 Memory Technology Comparison Table

```
┌──────────────┬───────────┬───────────┬───────────┬───────────┐
│   Property   │   SRAM    │   DRAM    │  NAND     │   HDD     │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ Access Time  │  1-2ns    │  50-100ns │  25us     │  5-10ms   │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ Cost/GB      │  ~$5000   │  ~$3-5    │  ~$0.1    │  ~$0.02   │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ Volatile     │    Yes    │    Yes    │    No     │    No     │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ Typical Size │  ~32MB    │  ~128GB   │  ~4TB     │  ~20TB    │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ Density      │    Low    │   High    │ Very High │ Very High │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ Power        │   High    │  Medium   │   Low     │   High    │
├──────────────┼───────────┼───────────┼───────────┼───────────┤
│ Main Use     │  Cache    │Main Memory│   SSD     │Mass Storage│
└──────────────┴───────────┴───────────┴───────────┴───────────┘
```

---

## 4. Memory Levels

### 4.1 Registers

```
Location: Inside CPU
Capacity: Tens to hundreds (x86-64: 16 general purpose + others)
Speed: 1 cycle (0.3ns @ 3GHz)

Register Types:
┌────────────────────────────────────────────────────────┐
│ General Purpose   │ RAX, RBX, RCX, RDX, RSI, RDI, ... │
├────────────────────────────────────────────────────────┤
│ Program Counter   │ RIP (Instruction Pointer)         │
├────────────────────────────────────────────────────────┤
│ Stack Pointer     │ RSP, RBP                          │
├────────────────────────────────────────────────────────┤
│ Flag Register     │ RFLAGS (Zero, Carry, Sign, ...)   │
├────────────────────────────────────────────────────────┤
│ Vector Registers  │ XMM0-15, YMM0-15, ZMM0-31 (SIMD)  │
├────────────────────────────────────────────────────────┤
│ Floating Point    │ ST0-ST7 (x87), XMM (SSE)          │
└────────────────────────────────────────────────────────┘

Register vs Memory Performance:
- Register operation: ADD R1, R2, R3  → 1 cycle
- Memory operation:   ADD R1, [mem]   → 4+ cycles (L1 hit)
```

### 4.2 Cache Memory

```
Cache Hierarchy (Modern CPU):

┌─────────────────────────────────────────────────────────────┐
│                          Core 0                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  ┌──────────────┐    ┌──────────────┐               │    │
│  │  │ L1 I-Cache   │    │ L1 D-Cache   │               │    │
│  │  │   32KB       │    │   32KB       │               │    │
│  │  │   4 cycles   │    │   4 cycles   │               │    │
│  │  └──────┬───────┘    └──────┬───────┘               │    │
│  │         │                   │                        │    │
│  │         └─────────┬─────────┘                        │    │
│  │                   │                                  │    │
│  │         ┌─────────┴─────────┐                        │    │
│  │         │     L2 Cache      │                        │    │
│  │         │      256KB        │                        │    │
│  │         │     12 cycles     │                        │    │
│  │         └─────────┬─────────┘                        │    │
│  └───────────────────┼──────────────────────────────────┘    │
│                      │                                       │
├──────────────────────┼───────────────────────────────────────┤
│                      │           Core 1                      │
│                      │     (same structure)                  │
├──────────────────────┼───────────────────────────────────────┤
│           ┌──────────┴──────────┐                            │
│           │     L3 Cache         │                            │
│           │   8-32MB (shared)    │                            │
│           │    40 cycles         │                            │
│           └──────────┬──────────┘                            │
└──────────────────────┼───────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Main Memory   │
              │   100+ cycles   │
              └─────────────────┘

Cache Characteristics Summary:
┌───────┬──────────┬───────────────┬──────────────────────────┐
│ Level │ Capacity │   Latency     │         Features         │
├───────┼──────────┼───────────────┼──────────────────────────┤
│  L1   │  32-64KB │   4 cycles    │ Per-core, I/D split      │
├───────┼──────────┼───────────────┼──────────────────────────┤
│  L2   │ 256KB-1MB│  12 cycles    │ Per-core or shared       │
├───────┼──────────┼───────────────┼──────────────────────────┤
│  L3   │  8-64MB  │  40 cycles    │ Shared by all cores      │
└───────┴──────────┴───────────────┴──────────────────────────┘
```

### 4.3 Main Memory (DRAM)

```
Modern Memory System Structure:

          ┌─────────────────────────────────────────────┐
          │               Memory Controller              │
          │          (integrated in CPU or northbridge)  │
          └──────────────────┬──────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
    │Channel 0│         │Channel 1│         │Channel 2│
    └────┬────┘         └────┬────┘         └────┬────┘
         │                   │                   │
    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
    │ DIMM 0  │         │ DIMM 0  │         │ DIMM 0  │
    │ 16GB    │         │ 16GB    │         │ 16GB    │
    └─────────┘         └─────────┘         └─────────┘

DDR (Double Data Rate) Evolution:
┌────────┬─────────────┬──────────────┬───────────────────┐
│ Gen    │ Clock Speed │ BW/Channel   │     Features      │
├────────┼─────────────┼──────────────┼───────────────────┤
│ DDR3   │  800-2133   │  6.4-17GB/s  │ 1.5V              │
├────────┼─────────────┼──────────────┼───────────────────┤
│ DDR4   │ 1600-3200   │ 12.8-25.6GB/s│ 1.2V, bank groups │
├────────┼─────────────┼──────────────┼───────────────────┤
│ DDR5   │ 3200-6400+  │ 25.6-51.2GB/s│ 1.1V, on-die ECC  │
└────────┴─────────────┴──────────────┴───────────────────┘
```

### 4.4 Secondary Storage

```
SSD Structure:
┌─────────────────────────────────────────────────────────────┐
│                        SSD Controller                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Flash Translation Layer (FTL)                        │    │
│  │ - Logical to physical address translation            │    │
│  │ - Wear Leveling                                     │    │
│  │ - Garbage Collection                                │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ NAND    │ │ NAND    │ │ NAND    │ │ NAND    │           │
│  │ Package │ │ Package │ │ Package │ │ Package │           │
│  │   0     │ │   1     │ │   2     │ │   3     │           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ NAND    │ │ NAND    │ │ NAND    │ │ NAND    │           │
│  │ Package │ │ Package │ │ Package │ │ Package │           │
│  │   4     │ │   5     │ │   6     │ │   7     │           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
└─────────────────────────────────────────────────────────────┘

HDD vs SSD Comparison:
┌─────────────────┬────────────────┬────────────────┐
│    Property     │      HDD       │      SSD       │
├─────────────────┼────────────────┼────────────────┤
│ Sequential Read │   150 MB/s     │  500-7000 MB/s │
├─────────────────┼────────────────┼────────────────┤
│ Random Read     │   0.5 MB/s     │  50-500 MB/s   │
├─────────────────┼────────────────┼────────────────┤
│ Latency         │   5-10 ms      │   0.02-0.1 ms  │
├─────────────────┼────────────────┼────────────────┤
│ IOPS (Random)   │    ~200        │  10K-1M        │
├─────────────────┼────────────────┼────────────────┤
│ Shock Resistance│    Weak        │    Strong      │
├─────────────────┼────────────────┼────────────────┤
│ Price per GB    │   $0.02        │   $0.10        │
└─────────────────┴────────────────┴────────────────┘
```

---

## 5. Memory Bandwidth and Latency

### 5.1 Bandwidth

```
Definition: Amount of data that can be transferred per unit time (bytes/second)

Calculation:
Bandwidth = Bus Width × Transfer Rate

Example (DDR4-3200):
- Bus width: 64 bits = 8 bytes
- Transfer rate: 3200 MT/s (Mega Transfers per second)
- Bandwidth = 8 × 3200 = 25,600 MB/s = 25.6 GB/s (per channel)

Dual channel: 25.6 × 2 = 51.2 GB/s
Quad channel: 25.6 × 4 = 102.4 GB/s

Memory Bandwidth by System:
┌─────────────────────────┬────────────────────────┐
│        System           │       Bandwidth        │
├─────────────────────────┼────────────────────────┤
│ Desktop (DDR4)          │    25-50 GB/s          │
├─────────────────────────┼────────────────────────┤
│ High-end Workstation    │   100-200 GB/s         │
├─────────────────────────┼────────────────────────┤
│ Server (8-ch DDR5)      │   300-400 GB/s         │
├─────────────────────────┼────────────────────────┤
│ GPU (HBM2)              │   1-2 TB/s             │
└─────────────────────────┴────────────────────────┘
```

### 5.2 Latency

```
Definition: Time from data request to receipt

Latency by Memory Level:

Level       │ Latency (cycles) │ Latency (ns) │ Relative Cost
────────────┼──────────────────┼──────────────┼───────────
Registers   │        1         │     ~0.3     │    1x
L1 Cache    │       4-5        │     ~1.5     │    4x
L2 Cache    │      12-14       │     ~4       │   12x
L3 Cache    │      40-50       │    ~15       │   40x
Main Memory │     100-300      │   ~60-100    │  200x
SSD         │   10,000-50,000  │   25-100us   │ 50,000x
HDD         │ 10,000,000+      │    5-10ms    │ 20,000,000x

Visualization:
┌───────────────────────────────────────────────────────────┐
│                                                           │
│ L1:    ●                                                 │
│ L2:    ●●●                                               │
│ L3:    ●●●●●●●●●●●●●                                      │
│ RAM:   ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● │
│                                                           │
│ SSD:   [...]                    (too long to display)    │
│ HDD:   [...]                    (even longer)            │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 5.3 Latency vs Bandwidth Relationship

```
Little's Law:
Concurrent Requests = Bandwidth × Latency

Meaning for Memory Systems:
- Multiple concurrent requests needed to utilize bandwidth
- Longer latency requires more concurrent requests

Example:
- Bandwidth: 50 GB/s
- Latency: 100 ns
- Cache line: 64 bytes

Required concurrent requests:
= (50 × 10^9) × (100 × 10^-9) / 64
= 78 concurrent requests

Solutions:
- Out-of-order execution to issue multiple loads simultaneously
- Prefetching to request data in advance
- Memory Level Parallelism (MLP) utilization
```

### 5.4 AMAT (Average Memory Access Time)

```
Formula:
AMAT = Hit Time + (Miss Rate × Miss Penalty)

Example:
- L1 hit time: 4 cycles
- L1 miss rate: 5%
- L2 hit time: 12 cycles
- L2 miss rate: 20% (of L1 misses)
- Memory access time: 200 cycles

Calculation:
AMAT = 4 + 0.05 × (12 + 0.20 × 200)
     = 4 + 0.05 × (12 + 40)
     = 4 + 0.05 × 52
     = 4 + 2.6
     = 6.6 cycles

AMAT for Multi-level Cache:
┌──────────────────────────────────────────────────────────┐
│ AMAT = T_L1 + MR_L1 × (T_L2 + MR_L2 × (T_L3 + MR_L3 × T_Mem))│
└──────────────────────────────────────────────────────────┘
```

---

## 6. Memory Performance Optimization

### 6.1 Data Layout Optimization

```
Structure of Arrays (SoA) vs Array of Structures (AoS):

AoS (Array of Structures):
// Why AoS hurts when accessing a single field: each Particle is 32 bytes, so
// consecutive x values are 32 bytes apart. A 64-byte cache line fetches only 2
// x values but also brings in 56 bytes of unused fields (y, z, vx, ...).
// Cache utilization is only 4/32 = 12.5%.
struct Particle {
    float x, y, z;      // 12 bytes
    float vx, vy, vz;   // 12 bytes
    float mass;         // 4 bytes
    int   id;           // 4 bytes
};                      // Total 32 bytes

Particle particles[1000];

Memory layout:
[x,y,z,vx,vy,vz,mass,id][x,y,z,vx,vy,vz,mass,id][...]

When accessing only x coordinates: Only 4 bytes used per 32 bytes → wasteful


SoA (Structure of Arrays):
// Why SoA is faster for field-wise access: all x values are contiguous in memory.
// A 64-byte cache line now holds 16 consecutive x values (16 × 4 bytes), giving
// 100% utilization. SIMD instructions (SSE/AVX) can process 4-8 floats per cycle
// from this packed layout — impossible with AoS's interleaved fields.
struct Particles {
    float x[1000];
    float y[1000];
    float z[1000];
    float vx[1000];
    float vy[1000];
    float vz[1000];
    float mass[1000];
    int   id[1000];
};

Memory layout:
[x0,x1,x2,...][y0,y1,y2,...][z0,z1,z2,...]...

When accessing only x coordinates: Contiguous memory access → efficient

Performance difference: SoA can be up to 4-8x faster (with SIMD)
```

### 6.2 Loop Optimization

```
Loop Tiling (Blocking):

// Original code - low cache efficiency
// Why the original is slow: the inner loop sweeps B[k][j] column-wise (stride = N),
// repeatedly evicting and reloading entire rows of B. For N=1024 with 8-byte doubles,
// the B matrix alone is 8 MB — far exceeding L1/L2 cache, causing thrashing.
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];

// With tiling - high cache efficiency
// Why tiling works: by processing BLOCK×BLOCK sub-matrices at a time, the working
// set (3 blocks of BLOCK² elements) fits in L1 cache. For BLOCK=64 with 8-byte
// doubles: 3 × 64² × 8 = 96 KB — comfortably within a 128 KB L1 data cache.
// Each element is reused BLOCK times before eviction, maximizing temporal locality.
#define BLOCK 64
for (ii = 0; ii < N; ii += BLOCK)
    for (jj = 0; jj < N; jj += BLOCK)
        for (kk = 0; kk < N; kk += BLOCK)
            for (i = ii; i < min(ii+BLOCK, N); i++)
                for (j = jj; j < min(jj+BLOCK, N); j++)
                    for (k = kk; k < min(kk+BLOCK, N); k++)
                        C[i][j] += A[i][k] * B[k][j];

Effect:
┌──────────────────────────────────────────────────────────┐
│ Original: Repeatedly accessing entire large matrix       │
│           → Cache thrashing                              │
│ Tiled: Completely process small blocks → Keep in cache  │
│                                                          │
│ N=1024, BLOCK=64:                                        │
│ - Original: ~10 seconds                                  │
│ - Tiled: ~2 seconds (5x improvement)                    │
└──────────────────────────────────────────────────────────┘
```

### 6.3 Prefetching

```
Hardware Prefetch:
- CPU detects access patterns and automatically prefetches
- Effective for sequential and strided access
- Ineffective for irregular access

Software Prefetch:
// Intel intrinsic example
// Why prefetch 16 elements ahead: if memory latency is ~80 cycles and each loop
// iteration takes ~5 cycles, we need to issue the prefetch 80/5 = 16 iterations
// early so the data arrives in L1 cache just in time for the actual load.
// _MM_HINT_T0 targets L1 (temporal); _MM_HINT_NTA targets non-temporal (streaming).
for (i = 0; i < N; i++) {
    _mm_prefetch(&array[i + 16], _MM_HINT_T0);  // Prefetch to L1
    sum += array[i];
}

Prefetch Distance Calculation:
Distance = Memory Latency / Time per Loop Iteration

Example:
- Memory latency: 100 cycles
- Time per loop iteration: 5 cycles
- Prefetch distance: 100 / 5 = 20 elements ahead

Cautions:
- Too early prefetch: Evicted from cache
- Too late prefetch: Data not arrived
- Unnecessary prefetch: Cache pollution
```

### 6.4 Memory Alignment

```
Importance of Alignment:

Aligned access:
Address: 0x1000 (64-byte aligned)
┌───────────────────────────────────────────────┐
│              64-byte cache line               │
│  [All 64 bytes of data fit in one line]       │
└───────────────────────────────────────────────┘
Access count: 1

Unaligned access:
Address: 0x1020 (32-byte offset)
┌─────────────────────┬─────────────────────────┐
│   Cache Line 1      │      Cache Line 2       │
│  [...32 bytes...]   │  [...32 bytes...]       │
└─────────────────────┴─────────────────────────┘
Access count: 2

Alignment Directives:
// C/C++
// Why align to 64 bytes: this matches the cache line size. If a struct straddles
// two cache lines (e.g., starts at byte 48 of one line), a single access requires
// loading TWO cache lines — doubling memory traffic and halving effective bandwidth.
// Aligning to the cache line boundary guarantees one-line access.
struct alignas(64) CacheLine {
    int data[16];
};

// Dynamic allocation
// Why use aligned_alloc instead of malloc: malloc only guarantees alignment to
// max_align_t (typically 16 bytes). For cache-line-aligned data (64 bytes) or
// SIMD-aligned data (32 bytes for AVX), explicit alignment is required.
void* ptr = aligned_alloc(64, size);
```

---

## 7. Practice Problems

### Basic Problems

1. Explain why memory hierarchy is necessary.

2. Identify temporal and spatial locality in the following code:
   ```c
   for (i = 0; i < 100; i++) {
       sum += array[i];
   }
   ```

3. What are 3 major differences between SRAM and DRAM?

### Intermediate Problems

4. If L1 cache hit time is 4 cycles, miss rate is 8%, and L2 access time is 12 cycles, what is the AMAT?

5. Which code harms spatial locality?
   ```c
   // (a)
   for (i = 0; i < N; i++)
       for (j = 0; j < N; j++)
           sum += a[i][j];

   // (b)
   for (j = 0; j < N; j++)
       for (i = 0; i < N; i++)
           sum += a[i][j];
   ```

6. What is the theoretical maximum bandwidth of DDR4-3200 dual channel?

### Advanced Problems

7. Explain the problems and solutions when Working Set size exceeds cache size.

8. Apply loop tiling to optimize the following matrix multiplication code:
   ```c
   for (i = 0; i < N; i++)
       for (j = 0; j < N; j++)
           for (k = 0; k < N; k++)
               C[i][j] += A[i][k] * B[k][j];
   ```

9. Calculate the prefetch distance:
   - Memory latency: 60ns
   - CPU clock: 3GHz
   - Instructions per loop: 10, CPI: 1

<details>
<summary>Answers</summary>

1. To bridge the speed gap between CPU and memory. By placing fast, expensive memory close to the CPU and slow, cheap memory farther away, we optimize performance relative to cost.

2. - Temporal locality: Variables `sum`, `i` (accessed 100 times)
   - Spatial locality: `array[i]` (accessing contiguous memory addresses)

3. SRAM vs DRAM:
   - Structure: SRAM 6T, DRAM 1T1C
   - Refresh: SRAM not needed, DRAM needed
   - Speed: SRAM fast (~2ns), DRAM slow (~50ns)
   - Cost/Density: SRAM expensive/low, DRAM cheap/high

4. AMAT = 4 + 0.08 × 12 = 4 + 0.96 = 4.96 cycles

5. (b) Column-major traversal - Arrays are stored row-major in C/C++

6. DDR4-3200: 8 bytes × 3200MT/s × 2 channels = 51.2 GB/s

7. Cache thrashing occurs - needed data is repeatedly replaced
   Solutions: Loop tiling, data layout optimization, algorithm changes

8. With tiling applied:
   ```c
   #define B 64
   for (ii = 0; ii < N; ii += B)
     for (jj = 0; jj < N; jj += B)
       for (kk = 0; kk < N; kk += B)
         for (i = ii; i < ii+B && i < N; i++)
           for (j = jj; j < jj+B && j < N; j++)
             for (k = kk; k < kk+B && k < N; k++)
               C[i][j] += A[i][k] * B[k][j];
   ```

9. Prefetch distance:
   - Memory latency: 60ns × 3GHz = 180 cycles
   - Loop time: 10 instructions × 1 CPI = 10 cycles
   - Distance: 180 / 10 = 18 elements ahead

</details>

---

## Next Steps

- [15_Cache_Memory.md](./15_Cache_Memory.md) - Cache mapping, replacement policies, write policies

---

## References

- Computer Architecture: A Quantitative Approach (Hennessy & Patterson)
- What Every Programmer Should Know About Memory (Ulrich Drepper)
- [Intel Memory Latency Checker](https://software.intel.com/content/www/us/en/develop/articles/intelr-memory-latency-checker.html)
- [Memory hierarchy visualization](https://www.youtube.com/watch?v=p3q5zWCw8J4)
