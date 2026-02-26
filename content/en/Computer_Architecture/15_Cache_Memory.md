# Cache Memory

**Previous**: [14_Memory_Hierarchy.md](./14_Memory_Hierarchy.md) | **Next**: [16_Virtual_Memory.md](./16_Virtual_Memory.md)

**Difficulty**: ⭐⭐⭐

**Prerequisites**: Memory Hierarchy, Principle of Locality

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how cache exploits the principle of locality
2. Compare direct-mapped, set-associative, and fully-associative cache mapping
3. Implement LRU and FIFO cache replacement policies
4. Distinguish write-through from write-back policies
5. Classify cache misses as compulsory, capacity, or conflict (the 3 Cs)
6. Explain multi-level cache organization (L1/L2/L3) and inclusion policies
7. Calculate cache performance metrics (hit rate, AMAT)

---

Cache memory is the single most important hardware optimization for software performance. A cache hit takes ~1 nanosecond; a cache miss that goes to main memory takes ~100 nanoseconds -- a 100x penalty. Understanding how caches work explains why matrix traversal order matters, why struct-of-arrays beats array-of-structs, and why your carefully optimized algorithm might still be slow if it has poor cache behavior.

## Table of Contents

1. [Cache Concept and Operation](#1-cache-concept-and-operation)
2. [Cache Mapping Schemes](#2-cache-mapping-schemes)
3. [Cache Replacement Policies](#3-cache-replacement-policies)
4. [Cache Write Policies](#4-cache-write-policies)
5. [Types of Cache Misses](#5-types-of-cache-misses)
6. [Multi-Level Cache](#6-multi-level-cache)
7. [Cache Optimization Techniques](#7-cache-optimization-techniques)
8. [Practice Problems](#8-practice-problems)

---

> Think of cache as your desk, main memory as a filing cabinet across the room, and disk as a warehouse across town. When you are working on a report, you keep the documents you need right on your desk (L1 cache) for instant access. Less frequently needed files go in the filing cabinet (main memory) — a short walk away. Rarely needed archives stay in the warehouse (disk) — retrieving them takes much longer. Cache design is the art of predicting which files you will need next and keeping them on your desk before you ask for them.

## 1. Cache Concept and Operation

### 1.1 What is Cache?

```
Cache Etymology: French "cacher" (to hide)

Definition: Small, high-speed memory located between CPU and main memory
Purpose: Keep frequently used data accessible quickly

┌─────────────────────────────────────────────────────────────┐
│                     Role of Cache                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│      CPU                   Cache               Main Memory  │
│   ┌───────┐           ┌───────────┐          ┌───────────┐  │
│   │       │  Fast     │           │  Slow    │           │  │
│   │       │◀────────▶│           │◀────────▶│           │  │
│   │       │  (~4 ns)  │           │ (~60 ns)  │           │  │
│   └───────┘           └───────────┘          └───────────┘  │
│                                                             │
│   Reduce average access time by keeping frequently used     │
│   data in cache                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Cache Structure Basics

```
Cache Line / Block:
- Basic unit of data transfer between cache and memory
- Typically 64 bytes (modern processors)

┌─────────────────────────────────────────────────────────────┐
│                      Cache Line (64 bytes)                   │
├─────────┬─────────────────────────────────────────────────┬──┤
│  Valid  │                     Tag                          │  │
│   (1b)  │                (upper address bits)              │  │
├─────────┴─────────────────────────────────────────────────┴──┤
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    Data (64 bytes)                     │  │
│  │  0x00 0x01 0x02 ... 0x3E 0x3F                         │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

Cache Terminology:
- Hit: Requested data is in cache
- Miss: Requested data is not in cache
- Hit Rate: Number of hits / Total accesses
- Miss Rate: 1 - Hit Rate
```

### 1.3 Address Decomposition

```
Decomposing memory address for cache access:

32-bit address example (4KB cache, 64B line, direct mapped):
┌─────────────────────────────────────────────────────────────┐
│ 31                    12 11          6 5               0    │
│ ├─────────────────────────┼─────────────┼────────────────┤  │
│ │         Tag (20b)       │ Index (6b)  │ Offset (6b)    │  │
│ └─────────────────────────┴─────────────┴────────────────┘  │
└─────────────────────────────────────────────────────────────┘

Offset (6 bits):
- Position within cache line
- 64 bytes = 2^6, thus 6 bits

Index (6 bits):
- Cache line selection
- 4KB / 64B = 64 lines = 2^6, thus 6 bits

Tag (20 bits):
- Distinguishes different addresses with same index
- Remaining bits = 32 - 6 - 6 = 20 bits
```

### 1.4 Cache Operation Flow

```
Cache Read Operation:

┌─────────────────────────────────────────────────────────────┐
│                      CPU Address Request                     │
│                           │                                 │
│                           ▼                                 │
│              ┌────────────────────────┐                     │
│              │ Select cache line by   │                     │
│              │      Index             │                     │
│              └────────────┬───────────┘                     │
│                           │                                 │
│                           ▼                                 │
│              ┌────────────────────────┐                     │
│              │ Compare Tag &          │                     │
│              │ Check Valid            │                     │
│              └────────────┬───────────┘                     │
│                           │                                 │
│              ┌────────────┴────────────┐                    │
│              │                         │                    │
│         Tag match &             Tag mismatch or             │
│         Valid = 1               Valid = 0                   │
│              │                         │                    │
│              ▼                         ▼                    │
│    ┌─────────────────┐      ┌─────────────────────┐        │
│    │   Cache Hit!    │      │    Cache Miss!      │        │
│    │ Select/return   │      │ Load block from     │        │
│    │ data by Offset  │      │ memory, store in    │        │
│    │                 │      │ cache, then return  │        │
│    └─────────────────┘      └─────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Cache Mapping Schemes

### 2.1 Direct Mapped

```
Characteristic: Each memory block can only be stored in one cache location

Calculation:
Cache Index = (Memory Block Address) mod (Number of Cache Lines)

Example: 8-line cache

Memory Block    Cache Index
   0      →      0
   1      →      1
   2      →      2
   ...
   7      →      7
   8      →      0  (conflict!)
   9      →      1
   ...
   16     →      0  (conflict!)

┌─────────────────────────────────────────────────────────────┐
│                     Direct Mapped Cache                      │
├─────────────────────────────────────────────────────────────┤
│  Index │ Valid │    Tag    │         Data (64B)            │
├────────┼───────┼───────────┼───────────────────────────────┤
│   0    │   1   │   0x100   │  [block 0 or 8 or 16...]      │
│   1    │   1   │   0x101   │  [block 1 or 9 or 17...]      │
│   2    │   0   │     -     │           -                   │
│   3    │   1   │   0x102   │  [block 3 or 11...]          │
│   4    │   1   │   0x103   │  [...]                        │
│   5    │   0   │     -     │           -                   │
│   6    │   1   │   0x100   │  [...]                        │
│   7    │   1   │   0x101   │  [...]                        │
└────────┴───────┴───────────┴───────────────────────────────┘

Advantages:
- Simple hardware
- Fast access (no parallel tag comparison needed)
- Deterministic replacement (no replacement policy needed)

Disadvantages:
- Many conflict misses
- Blocks mapping to same index compete
```

### 2.2 Fully Associative

```
Characteristic: Memory block can be stored in any cache location

┌─────────────────────────────────────────────────────────────┐
│                   Fully Associative Cache                    │
├─────────────────────────────────────────────────────────────┤
│  Entry │ Valid │    Tag       │         Data (64B)         │
├────────┼───────┼──────────────┼────────────────────────────┤
│   0    │   1   │  0x0001000   │  [any block]               │
│   1    │   1   │  0x0002008   │  [any block]               │
│   2    │   1   │  0x0001008   │  [any block]               │
│   3    │   1   │  0x0003010   │  [any block]               │
│   4    │   0   │      -       │      -                     │
│   5    │   1   │  0x0002000   │  [any block]               │
│   6    │   1   │  0x0004020   │  [any block]               │
│   7    │   1   │  0x0001020   │  [any block]               │
└────────┴───────┴──────────────┴────────────────────────────┘

Search Process:
┌─────────────────────────────────────────────────────────────┐
│  Compare request tag with all entry tags simultaneously     │
│  (parallel comparison)                                      │
│                                                             │
│  Request Tag: 0x0001008                                     │
│                                                             │
│  Entry 0: 0x0001000 ≠ 0x0001008 → Miss                     │
│  Entry 1: 0x0002008 ≠ 0x0001008 → Miss                     │
│  Entry 2: 0x0001008 = 0x0001008 → Hit!  ←                  │
│  Entry 3: 0x0003010 ≠ 0x0001008 → Miss                     │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘

Advantages:
- No conflict misses
- Maximum flexibility

Disadvantages:
- Complex hardware (parallel tag comparison needed)
- Slow access (requires CAM - Content Addressable Memory)
- Expensive implementation
- Practically used only for small structures like TLB
```

### 2.3 Set Associative

```
Characteristic: Compromise between direct mapped and fully associative
- Cache divided into multiple sets
- Fully associative within each set

N-way Set Associative:
- N cache lines (ways) per set
- Memory block can only map to one set
- Can be placed in any way within the set

Example: 4-way Set Associative (32 lines, 8 sets)

┌─────────────────────────────────────────────────────────────┐
│                  4-way Set Associative Cache                 │
├─────────────────────────────────────────────────────────────┤
│        │   Way 0   │   Way 1   │   Way 2   │   Way 3      │
│  Set   │ V│Tag│Data│ V│Tag│Data│ V│Tag│Data│ V│Tag│Data   │
├────────┼───────────┼───────────┼───────────┼──────────────┤
│   0    │ 1│100│... │ 1│108│... │ 0│ - │ -  │ 1│110│...    │
│   1    │ 1│101│... │ 1│109│... │ 1│111│... │ 0│ - │ -     │
│   2    │ 1│102│... │ 0│ - │ -  │ 1│10A│... │ 1│112│...    │
│   3    │ 1│103│... │ 1│10B│... │ 1│113│... │ 1│11B│...    │
│   4    │ 0│ - │ -  │ 1│10C│... │ 0│ - │ -  │ 1│114│...    │
│   5    │ 1│105│... │ 1│10D│... │ 1│115│... │ 0│ - │ -     │
│   6    │ 1│106│... │ 0│ - │ -  │ 1│10E│... │ 1│116│...    │
│   7    │ 1│107│... │ 1│10F│... │ 1│117│... │ 1│11F│...    │
└────────┴───────────┴───────────┴───────────┴──────────────┘

Address decomposition (32-bit, 4-way, 8 sets, 64B line):
┌─────────────────────────────────────────────────────────────┐
│ 31                   9  8         6  5                 0    │
│ ├───────────────────────┼───────────┼──────────────────┤    │
│ │     Tag (23 bits)     │Set Index  │  Block Offset    │    │
│ │                       │  (3 bits) │    (6 bits)      │    │
│ └───────────────────────┴───────────┴──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Mapping Scheme Comparison

```
┌─────────────────┬─────────────┬──────────────┬──────────────┐
│    Property     │Direct Mapped│Set Associative│Fully Assoc.  │
├─────────────────┼─────────────┼──────────────┼──────────────┤
│ Flexibility     │    Low      │    Medium    │    High      │
├─────────────────┼─────────────┼──────────────┼──────────────┤
│ HW Complexity   │    Low      │    Medium    │    High      │
├─────────────────┼─────────────┼──────────────┼──────────────┤
│ Access Speed    │    Fast     │    Medium    │    Slow      │
├─────────────────┼─────────────┼──────────────┼──────────────┤
│ Conflict Misses │    Many     │    Few       │    None      │
├─────────────────┼─────────────┼──────────────┼──────────────┤
│ # Comparators   │     1       │   N (N-way)  │  All lines   │
├─────────────────┼─────────────┼──────────────┼──────────────┤
│ Practical Use   │ Early cache │ Most caches  │    TLB       │
└─────────────────┴─────────────┴──────────────┴──────────────┘

Typical configurations (modern processors):
- L1 cache: 8-way set associative
- L2 cache: 4-8 way set associative
- L3 cache: 16-way set associative
- TLB: Fully associative or high associativity
```

---

## 3. Cache Replacement Policies

### 3.1 Need for Replacement Policy

```
In set associative or fully associative caches:
- New block must be loaded on cache miss
- When set (or entire cache) is full
- Decision needed on which block to replace

Goal: Replace block that won't be used in future → Minimize miss rate
Problem: Cannot predict future → Use heuristics
```

### 3.2 LRU (Least Recently Used)

```
Principle: Replace the block that was used longest ago
Rationale: Recently used blocks are likely to be used again soon (temporal locality)

Example: 4-way cache, access order A, B, C, D, E, A, B, F

Initial state (empty cache):
┌────┬────┬────┬────┐
│ -  │ -  │ -  │ -  │  LRU order: -
└────┴────┴────┴────┘

Access A (Miss):
┌────┬────┬────┬────┐
│ A  │ -  │ -  │ -  │  LRU order: A
└────┴────┴────┴────┘

Access B (Miss):
┌────┬────┬────┬────┐
│ A  │ B  │ -  │ -  │  LRU order: A, B
└────┴────┴────┴────┘

Access C (Miss):
┌────┬────┬────┬────┐
│ A  │ B  │ C  │ -  │  LRU order: A, B, C
└────┴────┴────┴────┘

Access D (Miss):
┌────┬────┬────┬────┐
│ A  │ B  │ C  │ D  │  LRU order: A, B, C, D
└────┴────┴────┴────┘

Access E (Miss, replace A - oldest):
┌────┬────┬────┬────┐
│ E  │ B  │ C  │ D  │  LRU order: B, C, D, E
└────┴────┴────┴────┘

Access A (Miss, replace B):
┌────┬────┬────┬────┐
│ E  │ A  │ C  │ D  │  LRU order: C, D, E, A
└────┴────┴────┴────┘

Access B (Miss, replace C):
┌────┬────┬────┬────┐
│ E  │ A  │ B  │ D  │  LRU order: D, E, A, B
└────┴────┴────┴────┘

LRU Implementation:
- Exact LRU: N! states needed → High hardware cost
- Approximate LRU: Bit matrix, tree-based, etc.
```

### 3.3 Pseudo-LRU (Tree-based)

```
Tree-based approximate LRU (4-way example):

          ┌───┐
          │ 0 │  ← Root: 0 means left more recent, 1 means right
          └─┬─┘
        ┌───┴───┐
      ┌─┴─┐   ┌─┴─┐
      │ 0 │   │ 1 │  ← Internal nodes
      └─┬─┘   └─┬─┘
     ┌──┴──┐ ┌──┴──┐
     │     │ │     │
    Way0  Way1 Way2 Way3

Finding way to replace:
1. Start from root
2. If bit is 0, go left; if 1, go right
3. At leaf, replace that way

Update bits on access:
- Set bits on path to accessed way in opposite direction
- Guides next replacement away from that way

Advantage: Only log2(N) bits needed (4-way: 3 bits vs exact LRU: more)
Disadvantage: Doesn't guarantee exact LRU order
```

### 3.4 FIFO (First In, First Out)

```
Principle: Replace the block that entered first

Example: 4-way cache, access order A, B, C, D, E, A

┌────┬────┬────┬────┐
│ -  │ -  │ -  │ -  │  Head=0
└────┴────┴────┴────┘

A (Miss) → B (Miss) → C (Miss) → D (Miss):
┌────┬────┬────┬────┐
│ A  │ B  │ C  │ D  │  Head=0 (A is first)
└────┴────┴────┴────┘

E (Miss, replace A):
┌────┬────┬────┬────┐
│ E  │ B  │ C  │ D  │  Head=1 (B is next)
└────┴────┴────┴────┘

A (Miss, replace B - even though A was just accessed!):
┌────┬────┬────┬────┐
│ E  │ A  │ C  │ D  │  Head=2
└────┴────┴────┴────┘

Advantages:
- Very simple implementation (one pointer)
- Fair replacement

Disadvantages:
- Ignores access patterns
- Belady's anomaly: Miss rate can increase with more cache
```

### 3.5 Random Replacement

```
Principle: Randomly select block to replace

Implementation:
- Use random number generator
- Or simple counter

┌────┬────┬────┬────┐
│ A  │ B  │ C  │ D  │
└────┴────┴────┴────┘

Insert E, random selection = Way 2:
┌────┬────┬────┬────┐
│ A  │ B  │ E  │ D  │  C replaced
└────┴────┴────┴────┘

Advantages:
- Very simple implementation
- Robust against pathological access patterns
- No Belady's anomaly

Disadvantages:
- Average performance lower than LRU
- Recently used blocks may be replaced

Practical Use:
- Some ARM processor caches
- Random+LRU hybrid in L2/L3 caches
```

### 3.6 Replacement Policy Comparison

```
Miss rate comparison (typical workloads):

┌────────────────┬───────────────┬───────────────────────────┐
│   Policy       │ Relative Miss │          Features         │
│                │     Rate      │                           │
├────────────────┼───────────────┼───────────────────────────┤
│ Optimal (OPT)  │    1.00x     │ Theoretical optimal       │
│                │              │ (requires future knowledge)│
├────────────────┼───────────────┼───────────────────────────┤
│ LRU            │    1.05x     │ Very good, complex impl.  │
├────────────────┼───────────────┼───────────────────────────┤
│ Pseudo-LRU     │    1.08x     │ LRU approximation,        │
│                │              │ efficient implementation   │
├────────────────┼───────────────┼───────────────────────────┤
│ Random         │    1.15x     │ Simple, reasonable perf.  │
├────────────────┼───────────────┼───────────────────────────┤
│ FIFO           │    1.20x     │ Simplest, lowest perf.    │
└────────────────┴───────────────┴───────────────────────────┘
```

---

## 4. Cache Write Policies

### 4.1 Write-Through

```
Principle: Update both cache and memory on write

┌─────────────────────────────────────────────────────────────┐
│                    Write-Through Operation                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│         CPU                                                 │
│          │                                                  │
│          │ Write X = 5                                      │
│          ▼                                                  │
│     ┌─────────┐                                             │
│     │  Cache  │ ───────────────────────┐                    │
│     │ X = 5   │                        │ X = 5             │
│     └─────────┘                        ▼                    │
│                                  ┌──────────┐               │
│                                  │  Memory  │               │
│                                  │  X = 5   │               │
│                                  └──────────┘               │
│                                                             │
│  Cache and memory always consistent                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Advantages:
- Cache and memory always consistent
- No write-back needed on cache replacement
- Simple implementation
- Easy data recovery (power failure, etc.)

Disadvantages:
- Memory access on every write → slow
- Consumes memory bandwidth

Mitigation: Write Buffer
┌──────────────────────────────────────────────┐
│     CPU → Cache → Write Buffer → Memory      │
│                     ↑                        │
│            CPU continues until buffer fills  │
└──────────────────────────────────────────────┘
```

### 4.2 Write-Back

```
Principle: Update only cache on write, update memory on replacement

┌─────────────────────────────────────────────────────────────┐
│                     Write-Back Operation                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Write occurs                                            │
│         CPU                                                 │
│          │                                                  │
│          │ Write X = 5                                      │
│          ▼                                                  │
│     ┌─────────┐                                             │
│     │  Cache  │       No memory access!                     │
│     │ X = 5   │◄── Dirty = 1                               │
│     └─────────┘                                             │
│                                                             │
│  2. On cache replacement                                    │
│     ┌─────────┐                                             │
│     │  Cache  │                                             │
│     │ X = 5   │ ─── If Dirty = 1, write to memory ───┐     │
│     └─────────┘                                      ▼     │
│                                             ┌──────────┐    │
│                                             │  Memory  │    │
│                                             │  X = 5   │    │
│                                             └──────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Dirty Bit:
- Indicates if cache line has been modified
- 1: Modified (inconsistent with memory)
- 0: Not modified (consistent with memory)

Advantages:
- Improved write performance (reduced memory access)
- Efficient for repeated writes to same address
- Saves memory bandwidth

Disadvantages:
- Complex implementation (dirty bit, write-back on replacement)
- Cache-memory inconsistency possible
- Delay on replacement
```

### 4.3 Write Allocate vs No-Write Allocate

```
How to handle write misses:

Write Allocate (Fetch on Write):
┌─────────────────────────────────────────────────────────────┐
│  Write miss occurs                                          │
│       ↓                                                     │
│  Load block from memory to cache                            │
│       ↓                                                     │
│  Perform write in cache                                     │
│                                                             │
│  Typically used with Write-Back                             │
│  Exploits spatial locality: adjacent data also loaded       │
└─────────────────────────────────────────────────────────────┘

No-Write Allocate (Write Around):
┌─────────────────────────────────────────────────────────────┐
│  Write miss occurs                                          │
│       ↓                                                     │
│  Write directly to memory (no cache load)                   │
│                                                             │
│  Typically used with Write-Through                          │
│  Efficient for data written once and not read               │
└─────────────────────────────────────────────────────────────┘

Common combinations:
┌──────────────────┬─────────────────┐
│  Write Policy    │  Allocate Policy│
├──────────────────┼─────────────────┤
│  Write-Back     │  Write Allocate │
│  Write-Through  │  No-Write Alloc │
└──────────────────┴─────────────────┘
```

### 4.4 Write Policy Comparison

```
┌─────────────────┬────────────────────┬────────────────────┐
│    Property     │   Write-Through    │    Write-Back      │
├─────────────────┼────────────────────┼────────────────────┤
│ Write Latency   │   High (memory)    │   Low (cache)      │
├─────────────────┼────────────────────┼────────────────────┤
│ Implementation  │       Low          │      High          │
│ Complexity      │                    │                    │
├─────────────────┼────────────────────┼────────────────────┤
│ Memory Traffic  │       High         │      Low           │
├─────────────────┼────────────────────┼────────────────────┤
│ Data Consistency│    Guaranteed      │   Needs management │
├─────────────────┼────────────────────┼────────────────────┤
│ Replacement     │       None         │  If dirty, yes     │
│ Overhead        │                    │                    │
├─────────────────┼────────────────────┼────────────────────┤
│ Practical Use   │  L1 (some),        │  L1, L2, L3        │
│                 │  Embedded          │                    │
└─────────────────┴────────────────────┴────────────────────┘
```

---

## 5. Types of Cache Misses

### 5.1 3C Classification

```
Three causes of cache misses (3C Model):

1. Cold Miss (Compulsory Miss)
2. Capacity Miss
3. Conflict Miss
```

### 5.2 Cold Miss (Compulsory Miss)

```
Definition: Miss on first access to a block
- Occurs when cache is empty
- Occurs regardless of cache size
- Unavoidable (can mitigate with prefetching)

┌─────────────────────────────────────────────────────────────┐
│  At program start:                                          │
│                                                             │
│  Access 1: A → Miss (first access to A)                     │
│  Access 2: B → Miss (first access to B)                     │
│  Access 3: A → Hit  (already in cache)                      │
│  Access 4: C → Miss (first access to C)                     │
│                                                             │
│  First 3 misses are Cold Misses                             │
└─────────────────────────────────────────────────────────────┘

Mitigation methods:
- Prefetching
- Larger block size (exploit spatial locality)
```

### 5.3 Capacity Miss

```
Definition: Miss due to insufficient cache capacity
- Occurs when Working Set > Cache size
- Previously cached block replaced due to capacity, then re-accessed

┌─────────────────────────────────────────────────────────────┐
│  4-block cache, access order: A, B, C, D, E, A              │
│                                                             │
│  Access A → Miss, cache: [A, -, -, -]                       │
│  Access B → Miss, cache: [A, B, -, -]                       │
│  Access C → Miss, cache: [A, B, C, -]                       │
│  Access D → Miss, cache: [A, B, C, D]  ← Cache full         │
│  Access E → Miss, cache: [E, B, C, D]  ← A replaced         │
│  Access A → Miss!                      ← Capacity Miss      │
│            cache: [E, A, C, D]        (A evicted due to     │
│                                        capacity)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Mitigation methods:
- Increase cache size
- Optimize data structures (reduce Working Set)
- Loop tiling and other algorithm optimizations
```

### 5.4 Conflict Miss

```
Definition: Miss due to blocks competing for same cache location
- Occurs in direct mapped or set associative caches
- Miss despite space available in cache due to location contention
- Does not occur in fully associative caches

┌─────────────────────────────────────────────────────────────┐
│  2-way cache, 2 sets, access order: A, E, I (all map to Set 0)│
│                                                             │
│  Initial: Set 0: [-, -], Set 1: [-, -]                     │
│                                                             │
│  Access A (Set 0) → Miss                                    │
│    Set 0: [A, -], Set 1: [-, -]                            │
│                                                             │
│  Access E (Set 0) → Miss                                    │
│    Set 0: [A, E], Set 1: [-, -]  ← Set 0 full              │
│                                                             │
│  Access I (Set 0) → Miss                                    │
│    Set 0: [I, E], Set 1: [-, -]  ← A replaced, Set 1 empty!│
│                                                             │
│  Access A (Set 0) → Miss! ← Conflict Miss                   │
│    Set 0: [I, A], Set 1: [-, -]  (Set 1 still empty)       │
│                                                             │
│  50% of cache empty but miss occurs = Conflict Miss         │
└─────────────────────────────────────────────────────────────┘

Mitigation methods:
- Increase associativity (increase N-way)
- Skewed cache
- Victim cache
```

### 5.5 3C Miss Proportions

```
Typical program miss proportions:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  │ 100%│ ████████████████████████████████████████          │
│  │     │ █      Compulsory (Cold)     █                    │
│  │     │ █████████████████████████████                     │
│  │  50%│ █     Capacity              █                     │
│  │     │ █████████████████████████████████████████         │
│  │     │ █           Conflict         █                    │
│  │   0%│                                                   │
│  └─────┴─────────────────────────────────────────          │
│        Small cache  ────────────────▶  Large cache         │
│                                                             │
│  - Increasing cache size: Reduces Capacity Miss            │
│  - Increasing associativity: Reduces Conflict Miss         │
│  - Cold Miss: Almost independent of cache size             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Multi-Level Cache

### 6.1 Need for Multi-Level Cache

```
Problem: L1 cache optimization dilemma
- Optimizing hit time → Small cache, low associativity
- Optimizing miss rate → Large cache, high associativity

Solution: Multi-level cache
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│        CPU  ← Fast access needed                            │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ L1 Cache│  Small and fast (hit time optimized)         │
│    │  32KB   │  4 cycles                                    │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ L2 Cache│  Medium (balanced)                           │
│    │  256KB  │  12 cycles                                   │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ L3 Cache│  Large and shared (miss rate optimized)      │
│    │  8-32MB │  40 cycles                                   │
│    └────┬────┘                                              │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │ Memory  │  100+ cycles                                 │
│    └─────────┘                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Inclusive vs Exclusive Cache

```
Inclusive Cache:
- Lower level contains data of upper levels
- All data in L1 is also in L2
- All data in L2 is also in L3

┌─────────────────────────────────────────────────────────────┐
│  L1: [A, B, C, D]                                           │
│  L2: [A, B, C, D, E, F, G, H, ...]                          │
│  L3: [A, B, C, D, E, F, G, H, ..., X, Y, Z]                 │
│                                                             │
│  Advantages:                                                │
│  - Easy to maintain cache coherence                         │
│  - Only need to check L3 when snooping                      │
│                                                             │
│  Disadvantages:                                             │
│  - Reduced effective capacity (duplicate storage)           │
│  - L3 replacement requires L1/L2 invalidation               │
└─────────────────────────────────────────────────────────────┘

Exclusive Cache:
- Each level stores different data
- A block exists in only one level

┌─────────────────────────────────────────────────────────────┐
│  L1: [A, B, C, D]                                           │
│  L2: [E, F, G, H]  (data not in L1)                         │
│  L3: [I, J, K, L]  (data not in L1 or L2)                   │
│                                                             │
│  Advantages:                                                │
│  - Total effective capacity = L1 + L2 + L3                  │
│                                                             │
│  Disadvantages:                                             │
│  - Complex data movement between caches                     │
│  - L1 miss requires sequential L2, L3 search                │
└─────────────────────────────────────────────────────────────┘

NINE (Non-Inclusive, Non-Exclusive):
- No consistency guarantee, freely placed
- Complex implementation but flexible
```

### 6.3 Private vs Shared Cache

```
Modern multicore processor structure:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│     Core 0          Core 1          Core 2          Core 3  │
│    ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐│
│    │ L1-I │        │ L1-I │        │ L1-I │        │ L1-I ││
│    │ L1-D │        │ L1-D │        │ L1-D │        │ L1-D ││
│    └──┬───┘        └──┬───┘        └──┬───┘        └──┬───┘│
│       │               │               │               │    │
│    ┌──┴───┐        ┌──┴───┐        ┌──┴───┐        ┌──┴───┐│
│    │  L2  │        │  L2  │        │  L2  │        │  L2  ││
│    │(priv)│        │(priv)│        │(priv)│        │(priv)││
│    └──┬───┘        └──┬───┘        └──┬───┘        └──┬───┘│
│       │               │               │               │    │
│       └───────────────┴───────┬───────┴───────────────┘    │
│                               │                             │
│                      ┌────────┴────────┐                    │
│                      │       L3        │                    │
│                      │    (shared)     │                    │
│                      │    8-32MB       │                    │
│                      └────────┬────────┘                    │
│                               │                             │
│                      ┌────────┴────────┐                    │
│                      │    Main Memory   │                    │
│                      └─────────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Private Cache (L1, L2):
- Dedicated to each core
- Minimizes access latency
- No inter-core interference

Shared Cache (L3):
- Shared by all cores
- Efficient for inter-core data sharing
- Dynamic capacity allocation (more space for busy cores)
- Requires coherence maintenance
```

### 6.4 Local vs Global Miss Rate

```
Local Miss Rate:
Miss rate at that level

Global Miss Rate:
Proportion of total accesses reaching that level

Example:
- L1 miss rate: 5%
- L2 miss rate (Local): 20%
- L3 miss rate (Local): 30%

Global miss rate calculation:
- L2 Global Miss Rate = L1 Miss Rate × L2 Local Miss Rate
                      = 0.05 × 0.20 = 1%
                      (1% of all accesses reach L3)

- L3 Global Miss Rate = L2 Global × L3 Local
                      = 0.01 × 0.30 = 0.3%
                      (0.3% of all accesses reach memory)

┌─────────────────────────────────────────────────────────────┐
│          Local vs Global Miss Rate                          │
├─────────────────────────────────────────────────────────────┤
│  Level  │  Local Miss Rate  │  Global Miss Rate            │
├─────────┼───────────────────┼─────────────────────────────-┤
│   L1    │       5%          │       5%                     │
│   L2    │      20%          │       1% (0.05 × 0.20)      │
│   L3    │      30%          │      0.3% (0.01 × 0.30)     │
└─────────┴───────────────────┴──────────────────────────────┘
```

---

## 7. Cache Optimization Techniques

### 7.1 Victim Cache

```
Purpose: Reduce Conflict Misses

Structure:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│      ┌─────────────────────────────────────┐                │
│      │        Main Cache (Direct Mapped)   │                │
│      └──────────────────┬──────────────────┘                │
│                         │                                   │
│                    Replaced block                           │
│                         │                                   │
│                         ▼                                   │
│      ┌─────────────────────────────────────┐                │
│      │    Victim Cache (Fully Associative) │                │
│      │         4-8 entries                 │                │
│      └─────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Operation:
1. Main Cache miss occurs
2. Search Victim Cache
3. If in Victim Cache → Swap and hit
4. If not → Load from memory, replaced block goes to Victim Cache

Effect: Absorbs conflict misses up to Victim Cache size
```

### 7.2 Prefetching

```
Hardware Prefetch:
- CPU detects access patterns
- Preloads blocks expected to be needed

Software Prefetch:
- Programmer/compiler inserts prefetch instructions

┌─────────────────────────────────────────────────────────────┐
│  // Software prefetch example                               │
│  for (int i = 0; i < N; i++) {                              │
│      // __builtin_prefetch(addr, rw, locality):             │
│      //   addr = address to prefetch                        │
│      //   rw: 0 = prefetch for read, 1 = prefetch for write │
│      //   locality: 0-3 temporal locality hint               │
│      //     3 = keep in all cache levels (high reuse)       │
│      //     0 = can evict immediately after use (streaming)  │
│      // 16 elements ahead ≈ 1 cache line of lead time:      │
│      //   64B line / 4B int = 16 ints per line, so prefetch │
│      //   starts loading the NEXT cache line just in time.   │
│      __builtin_prefetch(&a[i + 16], 0, 3);                  │
│      sum += a[i];                                           │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘

Prefetch effect:
┌─────────────────────────────────────────────────────────────┐
│  Without Prefetch:                                          │
│  Request ─────┬───── Memory Latency ─────┬───── Process    │
│                                                             │
│  With Prefetch:                                             │
│  Prefetch ────┬───── Memory Latency ─────┬                 │
│               │                          │                  │
│     Request ──┼───── Data arrives! ──────┼───── Process    │
│                                          │      (fast)      │
│  Prefetch hides memory latency                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Compiler Optimizations

```
Loop Interchange:
// C stores 2D arrays in row-major order: a[i][0], a[i][1], ... a[i][N-1] are
// contiguous in memory. A 64-byte cache line holds 16 ints, so sequential access
// along a row gets 15 hits for every 1 miss (spatial locality).

// Before: Column-major (poor locality)
// Iterating i in the inner loop jumps N*4 bytes between accesses (stride-N).
// If N > 16 (cache line / element size), every access misses — no spatial locality.
for (j = 0; j < N; j++)
    for (i = 0; i < N; i++)
        a[i][j] = b[i][j] + c[i][j];

// After: Row-major (good locality)
// Iterating j in the inner loop accesses consecutive memory addresses.
// Only 1 miss per 16 elements — 93.75% hit rate from spatial locality alone.
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
        a[i][j] = b[i][j] + c[i][j];


Loop Fusion:
// Before: Two loops — a[i] is written in loop 1 and read in loop 2.
// If the array is larger than the cache, a[i] is evicted before loop 2 reads it.
for (i = 0; i < N; i++) a[i] = b[i] + 1;
for (i = 0; i < N; i++) c[i] = a[i] * 2;

// After: One loop — a[i] is written and immediately read while still in L1 cache.
// This eliminates capacity misses for a[] and cuts total memory traffic roughly in half.
for (i = 0; i < N; i++) {
    a[i] = b[i] + 1;
    c[i] = a[i] * 2;
}


Loop Tiling:
// Before — naive matrix multiply.
// B[k][j] is accessed column-wise (stride-N): terrible cache behavior for large N.
// Working set = entire B matrix, which overflows cache when N is large.
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];

// After (block-wise processing for cache reuse)
// BLOCK is chosen so that a BLOCK×BLOCK sub-matrix of A, B, and C all fit in cache
// simultaneously (3 × BLOCK² × sizeof(double) ≤ L1 cache size).
// Within each tile, B is accessed with good locality because the working set is small.
for (ii = 0; ii < N; ii += BLOCK)
    for (jj = 0; jj < N; jj += BLOCK)
        for (kk = 0; kk < N; kk += BLOCK)
            for (i = ii; i < ii+BLOCK; i++)
                for (j = jj; j < jj+BLOCK; j++)
                    for (k = kk; k < kk+BLOCK; k++)
                        C[i][j] += A[i][k] * B[k][j];
```

---

## 8. Practice Problems

### Basic Problems

1. With a 32KB direct-mapped cache and 64-byte blocks, how many cache lines are there?

2. In a 4-way set associative cache, how many locations can a specific memory block be stored in?

3. Explain the difference between Write-Through and Write-Back.

### Intermediate Problems

4. Find the Tag, Index, and Offset for the following address:
   - Address: 0x12345678
   - Cache: 16KB, 2-way set associative, 64-byte lines

5. Classify the type of cache miss for the following access pattern:
   (4-line direct-mapped cache, blocks A~D all map to same index)
   ```
   A, B, C, D, A, E, A
   ```

6. L1 cache: 2-cycle hit time, 10% miss rate
   L2 cache: 10-cycle hit time, 5% miss rate
   Memory: 100-cycle access time
   Calculate the AMAT.

### Advanced Problems

7. Analyze the cache performance of the following code (64-byte cache line, 4-byte int):
   ```c
   int a[1024], b[1024], c[1024];
   // Sequential access pattern: each array is traversed linearly.
   // 64B line / 4B int = 16 ints per line → 1 cold miss per 16 accesses.
   // Hint: count total accesses, cold misses, and derive the hit rate.
   for (int i = 0; i < 1024; i++)
       c[i] = a[i] + b[i];
   ```

8. How many bits per set are needed to implement exact LRU policy in an 8-way set associative cache?

9. Explain the principle by which Victim Cache reduces Conflict Misses.

<details>
<summary>Answers</summary>

1. Number of cache lines = 32KB / 64B = 512 lines

2. 4 locations (4-way = 4 locations per set)

3. Write-Through: Updates cache and memory simultaneously, guarantees consistency
   Write-Back: Updates only cache, updates memory on replacement, better performance

4. Address decomposition:
   - 16KB, 2-way → 128 sets (16KB / 64B / 2 = 128)
   - Offset: 6 bits (64B = 2^6)
   - Index: 7 bits (128 = 2^7)
   - Tag: 32 - 6 - 7 = 19 bits

   0x12345678 = 0001 0010 0011 0100 0101 0110 0111 1000
   - Offset: 111000 (0x38)
   - Index: 1011001 (0x59)
   - Tag: 0001 0010 0011 0100 0101 (0x12345)

5. Miss classification:
   - A: Cold Miss (first access)
   - B: Cold Miss
   - C: Cold Miss
   - D: Cold Miss
   - A: Conflict Miss (replaced by D)
   - E: Conflict Miss + Cold Miss
   - A: Conflict Miss (replaced by E)

6. AMAT = 2 + 0.10 × (10 + 0.05 × 100)
        = 2 + 0.10 × 15
        = 2 + 1.5
        = 3.5 cycles

7. Cache analysis:
   - 64-byte line = 16 ints
   - Each array 1024 ints = 64 cache lines
   - Sequential access → 1 miss per 16 accesses
   - Cold Misses: 64 × 3 = 192 (for a, b, c each)
   - Hits: (1024 - 64) × 3 = 2880
   - Hit rate: 2880 / 3072 = 93.75%

8. Exact LRU implementation:
   - Representing order of 8 ways: 8! = 40320 states
   - Bits needed: log2(40320) ≈ 16 bits
   (In practice, approximate LRU is used)

9. Victim Cache principle:
   - Stores blocks replaced from Main Cache in Victim Cache
   - When a block replaced due to conflict is needed again, quickly found in Victim Cache
   - Small fully associative cache absorbs conflict misses

</details>

---

## Next Steps

- [16_Virtual_Memory.md](./16_Virtual_Memory.md) - Virtual address space, page tables, TLB

---

## References

- Computer Architecture: A Quantitative Approach (Hennessy & Patterson)
- [Cache Memory Visualization](https://www.ecs.umass.edu/ece/koren/architecture/Cache/frame1.htm)
- [Intel Optimization Manual - Cache](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- What Every Programmer Should Know About Memory (Ulrich Drepper)
