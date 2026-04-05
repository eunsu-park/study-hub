# Virtual Memory

**Previous**: [15_Cache_Memory.md](./15_Cache_Memory.md) | **Next**: [17_IO_Systems.md](./17_IO_Systems.md)

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: Memory Hierarchy, Cache Memory

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain virtual memory from the hardware perspective (MMU, page tables, TLB)
2. Describe how the TLB accelerates address translation
3. Trace a virtual-to-physical address translation through a multi-level page table
4. Explain how virtual memory provides memory protection between processes
5. Compare hardware-managed vs software-managed TLB designs
6. Analyze the interaction between cache and virtual memory (VIVT, PIPT, VIPT)

---

Virtual memory is where operating systems and hardware meet. The OS defines the page tables, but the hardware (MMU and TLB) performs the actual address translation on every single memory access -- billions of times per second. Understanding the hardware side of virtual memory explains why TLB misses are expensive, why huge pages improve performance, and how memory protection works at the circuit level.

## Table of Contents

1. [The Need for Virtual Memory](#1-the-need-for-virtual-memory)
2. [Address Spaces](#2-address-spaces)
3. [Pages and Page Frames](#3-pages-and-page-frames)
4. [Page Tables](#4-page-tables)
5. [TLB (Translation Lookaside Buffer)](#5-tlb-translation-lookaside-buffer)
6. [Page Faults and Page Replacement](#6-page-faults-and-page-replacement)
7. [Page Replacement Algorithms](#7-page-replacement-algorithms)
8. [Advanced Topics](#8-advanced-topics)
9. [Practice Problems](#9-practice-problems)

---

## 1. The Need for Virtual Memory

### 1.1 Problems Before Virtual Memory

```
Problem 1: Memory Capacity Limitation
┌─────────────────────────────────────────────────────────────┐
│  Program A: Needs 2GB                                        │
│  Physical Memory: 1GB                                        │
│                                                             │
│  → Cannot run Program A!                                    │
└─────────────────────────────────────────────────────────────┘

Problem 2: Memory Fragmentation
┌─────────────────────────────────────────────────────────────┐
│  Physical memory state:                                      │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐        │
│  │ Used │ Gap  │ Used │ Gap  │ Used │ Gap  │ Used │        │
│  │ 100M │ 50M  │ 200M │ 30M  │ 100M │ 70M  │ 150M │        │
│  └──────┴──────┴──────┴──────┴──────┴──────┴──────┘        │
│                                                             │
│  Total free space: 150MB, but no 100MB contiguous space     │
│  → Cannot load 100MB program (external fragmentation)       │
└─────────────────────────────────────────────────────────────┘

Problem 3: Lack of Memory Protection
┌─────────────────────────────────────────────────────────────┐
│  Program A can access Program B's memory region             │
│  → Security vulnerability, system instability               │
└─────────────────────────────────────────────────────────────┘

Problem 4: Relocation Problem
┌─────────────────────────────────────────────────────────────┐
│  Program compiled to run at specific address                 │
│  → Cannot run if that address is in use                     │
│  → Each program needs recompilation for different address   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Virtual Memory Solutions

```
┌─────────────────────────────────────────────────────────────┐
│                    Virtual Memory System                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Program A        Program B        Program C                │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐             │
│  │ Virtual │      │ Virtual │      │ Virtual │             │
│  │ Address │      │ Address │      │ Address │             │
│  │  Space  │      │  Space  │      │  Space  │             │
│  │  0~4GB  │      │  0~4GB  │      │  0~4GB  │             │
│  └────┬────┘      └────┬────┘      └────┬────┘             │
│       │                │                │                   │
│       └────────────────┼────────────────┘                   │
│                        │                                    │
│              ┌─────────┴─────────┐                          │
│              │    MMU (Memory     │                          │
│              │ Management Unit)   │                          │
│              │ Virtual→Physical   │                          │
│              │   Translation      │                          │
│              └─────────┬─────────┘                          │
│                        │                                    │
│              ┌─────────┴─────────┐                          │
│              │   Physical Memory  │                          │
│              │      (1GB, etc.)   │                          │
│              └─────────┬─────────┘                          │
│                        │                                    │
│              ┌─────────┴─────────┐                          │
│              │    Disk (Swap)    │                          │
│              │   (hundreds GB)   │                          │
│              └───────────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Advantages of Virtual Memory:
1. Capacity Extension: Use disk as memory extension
2. Memory Protection: Each process has independent address space
3. Memory Sharing: Multiple processes can share same physical pages
4. Fragmentation Solved: Non-contiguous physical memory appears contiguous
5. Relocation Transparency: All programs can start at same virtual address
```

---

## 2. Address Spaces

### 2.1 Virtual Address vs Physical Address

```
Virtual Address:
- Address used by programs
- Address generated by CPU
- Independent space for each process

Physical Address:
- Actual memory hardware address
- Address accessing DRAM chips
- Unique across entire system

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Process A            Process B            Physical Mem    │
│   Virtual Addr Space   Virtual Addr Space                   │
│                                                             │
│   0x00000000         0x00000000         0x00000000         │
│   ┌───────────┐      ┌───────────┐      ┌───────────┐      │
│   │   Code    │─────▶│   Code    │──┐   │  Frame 0  │      │
│   ├───────────┤      ├───────────┤  │   ├───────────┤      │
│   │   Data    │───┐  │   Data    │──┼──▶│  Frame 1  │      │
│   ├───────────┤   │  ├───────────┤  │   ├───────────┤      │
│   │   Heap    │   └─▶│   Heap    │──┼──▶│  Frame 2  │      │
│   ├───────────┤      ├───────────┤  │   ├───────────┤      │
│   │     ↓     │      │     ↓     │  └──▶│  Frame 3  │      │
│   │           │      │           │      ├───────────┤      │
│   │     ↑     │      │     ↑     │      │  Frame 4  │      │
│   ├───────────┤      ├───────────┤      ├───────────┤      │
│   │   Stack   │──────│   Stack   │─────▶│  Frame 5  │      │
│   └───────────┘      └───────────┘      └───────────┘      │
│   0xFFFFFFFF         0xFFFFFFFF         (actual size)      │
│                                                             │
│   Same virtual address 0x1000 can map to different physical│
│   addresses                                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Address Space Size

```
32-bit System:
- Virtual address space: 2^32 = 4GB
- Physical memory: Actual installed capacity (e.g., 4GB)

64-bit System:
- Theoretical virtual space: 2^64 = 16 exabytes
- Actual implementation (x86-64):
  - 48 bits used: 256TB virtual space
  - Physical address: Up to 52 bits supported (4PB)

┌─────────────────────────────────────────────────────────────┐
│        x86-64 Virtual Address Space Layout (48-bit)         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  0xFFFF_FFFF_FFFF_FFFF  ┌───────────────────┐              │
│                         │   Kernel Space    │              │
│                         │   (upper half)    │              │
│  0xFFFF_8000_0000_0000  ├───────────────────┤              │
│                         │                   │              │
│                         │   Non-Canonical   │ ← Not usable │
│                         │    (hole)         │              │
│                         │                   │              │
│  0x0000_7FFF_FFFF_FFFF  ├───────────────────┤              │
│                         │   User Space      │              │
│                         │   (lower half)    │              │
│  0x0000_0000_0000_0000  └───────────────────┘              │
│                                                             │
│  128TB available for each half                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Pages and Page Frames

### 3.1 Paging Concept

```
Page:
- Fixed-size unit dividing virtual address space
- Typically 4KB (4096 bytes)
- Management unit of virtual memory

Page Frame (Frame):
- Same-sized unit dividing physical memory
- Same size as page
- Management unit of physical memory

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│      Virtual Address Space          Physical Memory         │
│                                                             │
│    ┌─────────────┐           ┌─────────────┐               │
│    │   Page 0    │──────────▶│  Frame 3    │               │
│    ├─────────────┤           ├─────────────┤               │
│    │   Page 1    │──────┐    │  Frame 0    │               │
│    ├─────────────┤      │    ├─────────────┤               │
│    │   Page 2    │      └───▶│  Frame 1    │               │
│    ├─────────────┤           ├─────────────┤               │
│    │   Page 3    │──────────▶│  Frame 4    │               │
│    ├─────────────┤           ├─────────────┤               │
│    │   Page 4    │─ (Disk)   │  Frame 2    │◀─Other process│
│    ├─────────────┤           ├─────────────┤               │
│    │   Page 5    │──────────▶│  Frame 5    │               │
│    └─────────────┘           └─────────────┘               │
│                                                             │
│    - Page and Frame same size (4KB)                         │
│    - Mapping is arbitrary (non-contiguous possible)         │
│    - Some pages may be on disk                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Page Size

```
Common page sizes:
- 4KB (2^12): Most common
- 2MB (2^21): Large Page (Huge Page)
- 1GB (2^30): Gigantic Page

┌─────────────────────────────────────────────────────────────┐
│              Page Size Trade-offs                            │
├────────────────┬───────────────────┬────────────────────────┤
│                │  Small Page (4KB) │   Large Page (2MB)     │
├────────────────┼───────────────────┼────────────────────────┤
│ Internal Frag. │      Low          │       High             │
├────────────────┼───────────────────┼────────────────────────┤
│ Page Table     │      Large        │       Small            │
├────────────────┼───────────────────┼────────────────────────┤
│ TLB Coverage   │      Small        │       Large            │
├────────────────┼───────────────────┼────────────────────────┤
│ Disk Transfer  │    Inefficient    │      Efficient         │
├────────────────┼───────────────────┼────────────────────────┤
│ Suitable For   │  General programs │  Large memory apps     │
└────────────────┴───────────────────┴────────────────────────┘

Internal fragmentation example:
- Process needs 4097 bytes
- 4KB page: 2 pages needed (8KB), ~4KB wasted
- 2MB page: 1 page needed (2MB), ~2MB wasted
```

### 3.3 Virtual Address Decomposition

```
4KB page, 32-bit virtual address:

┌─────────────────────────────────────────────────────────────┐
│  31                              12  11                   0 │
│  ├────────────────────────────────┼────────────────────────┤│
│  │    Virtual Page Number (VPN)   │    Page Offset         ││
│  │         (20 bits)              │     (12 bits)          ││
│  └────────────────────────────────┴────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

Page Offset: 12 bits = Position within 4KB page (0-4095)
VPN: 20 bits = 2^20 = ~1 million virtual pages

Example: Virtual address 0x12345678
- VPN = 0x12345 (upper 20 bits)
- Offset = 0x678 (lower 12 bits)

Physical address after translation:
- Page table lookup: VPN 0x12345 → PFN 0x00ABC
- Physical address = 0x00ABC000 + 0x678 = 0x00ABC678

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Virtual addr:  0x12345  │  0x678                           │
│                  (VPN)    │ (Offset)                        │
│                   ↓                                         │
│            [Page Table]                                     │
│                   ↓                                         │
│  Physical addr: 0x00ABC  │  0x678                           │
│                  (PFN)    │ (Offset) ← Copied directly      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Page Tables

### 4.1 Single-Level Page Table

```
Structure:
┌─────────────────────────────────────────────────────────────┐
│                    Page Table                                │
├──────┬───────┬───────┬───────┬──────────────────────────────┤
│Index │ Valid │ PFN   │ Prot  │ Other Flags                  │
├──────┼───────┼───────┼───────┼──────────────────────────────┤
│  0   │   1   │ 0x003 │  RW   │ Present, Dirty               │
│  1   │   1   │ 0x001 │  R    │ Present                      │
│  2   │   0   │   -   │   -   │ Not Present (Disk)           │
│  3   │   1   │ 0x004 │  RW   │ Present                      │
│  4   │   1   │ 0x002 │  RWX  │ Present                      │
│ ...  │  ...  │  ...  │  ...  │ ...                          │
└──────┴───────┴───────┴───────┴──────────────────────────────┘

Page Table Entry (PTE) Structure (x86, 32-bit):
┌─────────────────────────────────────────────────────────────┐
│  31              12  11    9    8   7   6   5   4   3  2  1  0│
│  ├────────────────┼────────┼────┼───┼───┼───┼───┼───┼──┼──┼──┤│
│  │  Page Frame    │Reserved│ G  │PAT│ D │ A │PCD│PWT│U │W │P ││
│  │   Number       │        │    │   │   │   │   │   │/S│/R│  ││
│  └────────────────┴────────┴────┴───┴───┴───┴───┴───┴──┴──┴──┘│
└─────────────────────────────────────────────────────────────┘

P (Present): Page is in physical memory
W/R: Writable (1) / Read-only (0)
U/S: User accessible (1) / Kernel only (0)
A (Accessed): Has been accessed (read or write)
D (Dirty): Has been modified (written)

Problem: Memory waste
- 32-bit address, 4KB page: 2^20 = 1M entries
- 4 bytes per entry: 4MB page table per process
- Full table needed even if most space unused
```

### 4.2 Multi-Level Page Table

```
2-Level Page Table (32-bit, 4KB page):

┌─────────────────────────────────────────────────────────────┐
│  31                22  21              12  11             0 │
│  ├──────────────────┼──────────────────┼──────────────────┤ │
│  │  Page Dir Index  │  Page Table Index │   Page Offset    │ │
│  │    (10 bits)     │    (10 bits)      │    (12 bits)     │ │
│  └──────────────────┴──────────────────┴──────────────────┘ │
└─────────────────────────────────────────────────────────────┘

Translation Process:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Virtual Address: [Dir Index | Table Index | Offset]       │
│                         │            │           │          │
│                         ▼            │           │          │
│   ┌─────────────────────────────┐   │           │          │
│   │    Page Directory (4KB)     │   │           │          │
│   │    1024 entries × 4B        │   │           │          │
│   │  ┌─────────────────────┐    │   │           │          │
│   │  │  Entry 0 → PT0      │    │   │           │          │
│   │  │  Entry 1 → PT1      │    │   │           │          │
│   │  │  Entry 2 → null     │←── Unused regions have no table│
│   │  │  ...                │    │   │           │          │
│   │  │  Entry N → PTN      │────┼───┘           │          │
│   │  └─────────────────────┘    │               │          │
│   └─────────────────────────────┘               │          │
│                  │                              │          │
│                  ▼                              │          │
│   ┌─────────────────────────────┐              │          │
│   │    Page Table N (4KB)       │              │          │
│   │    1024 entries × 4B        │              │          │
│   │  ┌─────────────────────┐    │              │          │
│   │  │  Entry 0 → Frame X  │    │              │          │
│   │  │  Entry M → Frame Y  │────┼──────────────┘          │
│   │  │  ...                │    │                          │
│   │  └─────────────────────┘    │              │          │
│   └─────────────────────────────┘              │          │
│                  │                              │          │
│                  ▼                              │          │
│        Physical Frame Y + Offset = Physical Address        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Advantage:
- No page table allocation for unused regions
- Large savings if most virtual space is empty
```

### 4.3 4-Level Page Table (x86-64)

```
4-Level Page Table for 64-bit systems:

┌─────────────────────────────────────────────────────────────┐
│  63    48 47    39 38    30 29    21 20    12 11         0 │
│  ├──────┼────────┼────────┼────────┼────────┼────────────┤ │
│  │ Sign │  PML4  │  PDPT  │   PD   │   PT   │   Offset   │ │
│  │ Ext  │(9bits) │(9bits) │(9bits) │(9bits) │  (12bits)  │ │
│  └──────┴────────┴────────┴────────┴────────┴────────────┘ │
└─────────────────────────────────────────────────────────────┘

PML4: Page Map Level 4
PDPT: Page Directory Pointer Table
PD: Page Directory
PT: Page Table

Translation Process:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   CR3 Register                                              │
│        │                                                    │
│        ▼                                                    │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐            │
│   │  PML4   │─────▶│  PDPT   │─────▶│   PD    │            │
│   │ (512E)  │      │ (512E)  │      │ (512E)  │            │
│   └─────────┘      └─────────┘      └─────────┘            │
│                                          │                  │
│                                          ▼                  │
│                                    ┌─────────┐              │
│                                    │   PT    │              │
│                                    │ (512E)  │              │
│                                    └────┬────┘              │
│                                         │                   │
│                                         ▼                   │
│                              ┌─────────────────┐           │
│                              │  Physical Frame │           │
│                              │   + Offset      │           │
│                              └─────────────────┘           │
│                                                             │
│  Each level: 512 entries × 8 bytes = 4KB                   │
│  Maximum 4 memory accesses (on TLB miss)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Inverted Page Table

```
Traditional Page Table:
- Entries for all virtual pages
- Separate table per process

Inverted Page Table:
- Entries for physical frames only
- One table for entire system

┌─────────────────────────────────────────────────────────────┐
│              Inverted Page Table                             │
├───────┬────────┬──────────────┬─────────────────────────────┤
│ Frame │  PID   │     VPN      │         Chain               │
├───────┼────────┼──────────────┼─────────────────────────────┤
│   0   │   42   │   0x12345    │         null                │
│   1   │   17   │   0x00ABC    │         → 5                 │
│   2   │   42   │   0x67890    │         null                │
│   3   │   25   │   0x11111    │         null                │
│   4   │   17   │   0x00DEF    │         null                │
│   5   │   33   │   0x00ABC    │         null (hash chain)   │
│  ...  │  ...   │     ...      │         ...                 │
└───────┴────────┴──────────────┴─────────────────────────────┘

Search: Hash (PID, VPN) and search table
- Follow chain on hash collision
- Advantage: Memory savings (proportional to physical memory)
- Disadvantage: Search time (hash + chain)
```

---

## 5. TLB (Translation Lookaside Buffer)

### 5.1 Need for TLB

```
Page table access overhead:
- 4-level page table: 4 additional memory accesses
- 4 table lookups before actual data access

┌─────────────────────────────────────────────────────────────┐
│  Memory access without TLB:                                 │
│                                                             │
│  1. Access PML4 (memory)                                   │
│  2. Access PDPT (memory)                                   │
│  3. Access PD (memory)                                     │
│  4. Access PT (memory)                                     │
│  5. Access actual data (memory)                            │
│                                                             │
│  Total 5 memory accesses! (400+ cycles)                    │
│                                                             │
│  With TLB hit:                                             │
│  1. TLB lookup (~1 cycle)                                  │
│  2. Access actual data (memory)                            │
│                                                             │
│  Total ~100 cycles (TLB lookup + memory)                   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 TLB Structure

```
TLB (Translation Lookaside Buffer):
- Caches recent translation results
- Fully associative or set associative structure
- Very small (32-1024 entries)
- Very fast (~1 cycle)

┌─────────────────────────────────────────────────────────────┐
│                         TLB                                  │
├──────┬───────┬──────────┬──────────┬────────────────────────┤
│ ASID │ Valid │   VPN    │   PFN    │    Permissions         │
├──────┼───────┼──────────┼──────────┼────────────────────────┤
│  42  │   1   │ 0x12345  │ 0x00ABC  │    R/W, User           │
│  42  │   1   │ 0x12346  │ 0x00ABD  │    R/W, User           │
│  17  │   1   │ 0x00100  │ 0x00300  │    R, User             │
│  42  │   1   │ 0x7FFFF  │ 0x00123  │    R/W, User           │
│  17  │   1   │ 0x00200  │ 0x00400  │    R/W/X, User         │
│ ...  │  ...  │   ...    │   ...    │    ...                 │
└──────┴───────┴──────────┴──────────┴────────────────────────┘

ASID (Address Space ID):
- Distinguishes processes
- Avoids TLB flush on context switch
```

### 5.3 TLB Operation

```
Address Translation Process:

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  CPU generates virtual address                              │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │  TLB Lookup     │ ← Done in parallel (VPN extract & TLB) │
│  └────────┬────────┘                                        │
│           │                                                 │
│     ┌─────┴─────┐                                          │
│     │           │                                          │
│   Hit?        Miss                                         │
│     │           │                                          │
│     ▼           ▼                                          │
│  ┌──────┐   ┌─────────────────────────────────────┐        │
│  │ PFN  │   │  Page Table Walk                     │        │
│  │obtain│   │  (done by HW or SW)                  │        │
│  └──┬───┘   │  1. Access PML4                      │        │
│     │       │  2. Access PDPT                      │        │
│     │       │  3. Access PD                        │        │
│     │       │  4. Access PT                        │        │
│     │       │  5. Store result in TLB              │        │
│     │       └───────────────┬─────────────────────┘        │
│     │                       │                              │
│     └───────────┬───────────┘                              │
│                 │                                          │
│                 ▼                                          │
│     Physical address = PFN + Offset                        │
│                 │                                          │
│                 ▼                                          │
│     Cache/Memory access                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 TLB Performance

```
TLB Hit Rate and AMAT:

Typical TLB hit rate: 99% or higher

AMAT Calculation:
AMAT = TLB_Hit_Rate × (TLB_Time + Mem_Time) +
       TLB_Miss_Rate × (TLB_Time + Walk_Time + Mem_Time)

Example:
- TLB access: 1 cycle
- Page walk: 100 cycles (4 memory accesses)
- Memory access: 100 cycles
- TLB hit rate: 99%

TLB hit: 1 + 100 = 101 cycles
TLB miss: 1 + 100 + 100 = 201 cycles

AMAT = 0.99 × 101 + 0.01 × 201
     = 99.99 + 2.01
     = 102 cycles

Without TLB: 500 cycles (5 memory accesses)
→ TLB provides ~5x performance improvement

TLB Coverage:
- 64-entry TLB, 4KB pages: 256KB
- 64-entry TLB, 2MB pages: 128MB
- Large pages increase TLB efficiency
```

---

## 6. Page Faults and Page Replacement

### 6.1 Page Fault

```
Definition: Situation where accessed page is not in physical memory

Page Fault Causes:
1. Page is on disk (swapped out)
2. Accessing unallocated address
3. Permission violation (write to read-only page, etc.)

┌─────────────────────────────────────────────────────────────┐
│                  Page Fault Handling Process                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. CPU accesses virtual address                            │
│           │                                                 │
│           ▼                                                 │
│  2. MMU checks PTE: Present = 0                            │
│           │                                                 │
│           ▼                                                 │
│  3. Page Fault Exception raised                            │
│           │                                                 │
│           ▼                                                 │
│  4. OS Page Fault Handler executes                         │
│           │                                                 │
│     ┌─────┴─────────────────────────────────┐              │
│     │                                       │              │
│     ▼                                       ▼              │
│  Valid access?                          Invalid access      │
│     │                                       │              │
│     ▼                                       ▼              │
│  5. Find free frame                      Raise SIGSEGV     │
│     (replace page if none)               (terminate proc)  │
│           │                                                 │
│           ▼                                                 │
│  6. Read page from disk (~10ms)                            │
│           │                                                 │
│           ▼                                                 │
│  7. Update page table (Present = 1)                        │
│           │                                                 │
│           ▼                                                 │
│  8. Re-execute instruction                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Page Fault Cost

```
Page Fault Time:
- Disk read: ~10ms (HDD) or ~0.1ms (SSD)
- CPU clock: 3GHz = 0.33ns/cycle

Cycle count:
- HDD: 10ms / 0.33ns = 30,000,000 cycles
- SSD: 0.1ms / 0.33ns = 300,000 cycles

Normal memory access: ~100 cycles

Performance Impact Calculation:
For page fault rate p:
EAT = (1-p) × 100ns + p × 10ms
    = 100ns + p × 10,000,000ns

To limit performance degradation to 10%:
110ns ≥ 100ns + p × 10,000,000ns
p ≤ 0.000001 (one in a million)

→ Page fault rate must be very low!
```

### 6.3 Demand Paging vs Prepaging

```
Demand Paging:
- Load pages only when needed
- Low memory usage initially
- Page fault on first access

┌─────────────────────────────────────────────────────────────┐
│  At process start:                                          │
│  - Code pages not loaded                                    │
│  - Page fault on first instruction                          │
│  - Only needed pages loaded                                 │
└─────────────────────────────────────────────────────────────┘

Prepaging:
- Preload related pages
- Reduced initial page faults
- May load unused pages

┌─────────────────────────────────────────────────────────────┐
│  At process start:                                          │
│  - Predict and preload Working Set                          │
│  - Preload all or part of code region                       │
│  - Fewer initial page faults                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Page Replacement Algorithms

### 7.1 FIFO (First-In, First-Out)

```
Principle: Replace the page that entered first

Example: 3 frames, access order 1,2,3,4,1,2,5,1,2,3,4,5

┌────┬─────────┬─────────┬─────────┬───────┐
│Ref │ Frame 0 │ Frame 1 │ Frame 2 │Result │
├────┼─────────┼─────────┼─────────┼───────┤
│ 1  │    1    │    -    │    -    │ Miss  │
│ 2  │    1    │    2    │    -    │ Miss  │
│ 3  │    1    │    2    │    3    │ Miss  │
│ 4  │    4    │    2    │    3    │ Miss  │ ← 1 replaced
│ 1  │    4    │    1    │    3    │ Miss  │ ← 2 replaced
│ 2  │    4    │    1    │    2    │ Miss  │ ← 3 replaced
│ 5  │    5    │    1    │    2    │ Miss  │ ← 4 replaced
│ 1  │    5    │    1    │    2    │ Hit   │
│ 2  │    5    │    1    │    2    │ Hit   │
│ 3  │    3    │    1    │    2    │ Miss  │ ← 5 replaced
│ 4  │    3    │    4    │    2    │ Miss  │ ← 1 replaced
│ 5  │    3    │    4    │    5    │ Miss  │ ← 2 replaced
└────┴─────────┴─────────┴─────────┴───────┘

Total misses: 10, Hits: 2

Belady's Anomaly:
- In FIFO, miss rate can increase with more frames
- Counter-intuitive phenomenon
```

### 7.2 Optimal (OPT)

```
Principle: Replace the page that won't be used for longest time

Example: 3 frames, access order 1,2,3,4,1,2,5,1,2,3,4,5

┌────┬─────────┬─────────┬─────────┬───────┬────────────────┐
│Ref │ Frame 0 │ Frame 1 │ Frame 2 │Result │ Future refs    │
├────┼─────────┼─────────┼─────────┼───────┼────────────────┤
│ 1  │    1    │    -    │    -    │ Miss  │                │
│ 2  │    1    │    2    │    -    │ Miss  │                │
│ 3  │    1    │    2    │    3    │ Miss  │                │
│ 4  │    1    │    2    │    4    │ Miss  │ 3 used latest  │
│ 1  │    1    │    2    │    4    │ Hit   │                │
│ 2  │    1    │    2    │    4    │ Hit   │                │
│ 5  │    1    │    2    │    5    │ Miss  │ 4 not used again│
│ 1  │    1    │    2    │    5    │ Hit   │                │
│ 2  │    1    │    2    │    5    │ Hit   │                │
│ 3  │    1    │    2    │    3    │ Miss  │ 5 used latest  │
│ 4  │    1    │    4    │    3    │ Miss  │ 2 used latest  │
│ 5  │    1    │    4    │    5    │ Miss  │ 3 not used again│
└────┴─────────┴─────────┴─────────┴───────┴────────────────┘

Total misses: 8, Hits: 4

Optimal but not implementable (requires future prediction)
→ Used only as benchmark
```

### 7.3 LRU (Least Recently Used)

```
Principle: Replace the page used longest ago

Example: 3 frames, access order 1,2,3,4,1,2,5,1,2,3,4,5

┌────┬─────────┬─────────┬─────────┬───────┬────────────────┐
│Ref │ Frame 0 │ Frame 1 │ Frame 2 │Result │ LRU order      │
├────┼─────────┼─────────┼─────────┼───────┼────────────────┤
│ 1  │    1    │    -    │    -    │ Miss  │ 1              │
│ 2  │    1    │    2    │    -    │ Miss  │ 1,2            │
│ 3  │    1    │    2    │    3    │ Miss  │ 1,2,3          │
│ 4  │    4    │    2    │    3    │ Miss  │ 2,3,4 (1 repl) │
│ 1  │    4    │    1    │    3    │ Miss  │ 3,4,1 (2 repl) │
│ 2  │    4    │    1    │    2    │ Miss  │ 4,1,2 (3 repl) │
│ 5  │    5    │    1    │    2    │ Miss  │ 1,2,5 (4 repl) │
│ 1  │    5    │    1    │    2    │ Hit   │ 2,5,1          │
│ 2  │    5    │    1    │    2    │ Hit   │ 5,1,2          │
│ 3  │    3    │    1    │    2    │ Miss  │ 1,2,3 (5 repl) │
│ 4  │    3    │    4    │    2    │ Miss  │ 2,3,4 (1 repl) │
│ 5  │    3    │    4    │    5    │ Miss  │ 3,4,5 (2 repl) │
└────┴─────────┴─────────┴─────────┴───────┴────────────────┘

Total misses: 10, Hits: 2

Implementation methods:
1. Timestamp: Record last access time for each page
2. Stack: Move page to stack top on access
3. Approximate LRU: Clock algorithm, etc.
```

### 7.4 Clock Algorithm (Second Chance)

```
Principle: FIFO + Reference Bit to approximate LRU

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│          Clock Hand                                         │
│               ↓                                             │
│          ┌────────┐                                         │
│          │Page 0  │                                         │
│          │ R=0    │ ← If R=0, replace                       │
│       ┌──┴────────┴──┐                                      │
│       │              │                                      │
│   ┌───┴───┐      ┌───┴───┐                                  │
│   │Page 3 │      │Page 1 │                                  │
│   │ R=1   │      │ R=1   │ ← If R=1, set R=0 and skip      │
│   └───┬───┘      └───┬───┘                                  │
│       │              │                                      │
│       └──┬────────┬──┘                                      │
│          │Page 2  │                                         │
│          │ R=1    │                                         │
│          └────────┘                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Algorithm:
1. Check page at clock hand
2. If R=0, replace that page
3. If R=1, set R=0 and move to next page
4. If all pages have R=1, full circle resets all and replaces R=0 page

Enhanced Clock (NRU):
- Add Modified(M) bit
- Priority: (R=0,M=0) > (R=0,M=1) > (R=1,M=0) > (R=1,M=1)
- Modified pages require disk write, so lower priority
```

### 7.5 Algorithm Comparison

```
┌────────────────┬──────────────────┬─────────────────────────┐
│   Algorithm    │     Advantages   │      Disadvantages      │
├────────────────┼──────────────────┼─────────────────────────┤
│ FIFO           │ Simple impl.     │ Belady's Anomaly       │
│                │                  │ Unstable performance    │
├────────────────┼──────────────────┼─────────────────────────┤
│ Optimal        │ Optimal perf.    │ Not implementable       │
│                │ Benchmark        │ (requires prediction)   │
├────────────────┼──────────────────┼─────────────────────────┤
│ LRU            │ Good performance │ High impl. cost         │
│                │ Stack property   │ Hardware support needed │
├────────────────┼──────────────────┼─────────────────────────┤
│ Clock          │ Efficient approx.│ Slightly lower than LRU │
│                │ Easy impl.       │                         │
├────────────────┼──────────────────┼─────────────────────────┤
│ Working Set    │ Prevents thrash  │ Hard to set parameters  │
│                │ Adaptive         │                         │
└────────────────┴──────────────────┴─────────────────────────┘
```

---

## 8. Advanced Topics

### 8.1 Copy-on-Write (COW)

```
Principle: Share pages on fork() instead of copying

┌─────────────────────────────────────────────────────────────┐
│                     Copy-on-Write                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Right after fork():                                        │
│  Parent                 Child                               │
│  ┌─────────┐           ┌─────────┐                          │
│  │ Page 0  │──────┬────│ Page 0  │                          │
│  │ (R/O)   │      │    │ (R/O)   │                          │
│  ├─────────┤      │    ├─────────┤                          │
│  │ Page 1  │──┐   │ ┌──│ Page 1  │                          │
│  │ (R/O)   │  │   │ │  │ (R/O)   │                          │
│  └─────────┘  │   │ │  └─────────┘                          │
│               │   │ │                                       │
│               │   ▼ │                                       │
│               │ ┌─────┐                                     │
│               │ │Frame│  Physical Memory                    │
│               │ │  A  │  (shared)                           │
│               │ └─────┘                                     │
│               │   ▲                                         │
│               └───┘                                         │
│                                                             │
│  When Child writes to Page 1:                               │
│  ┌─────────┐           ┌─────────┐                          │
│  │ Page 1  │           │ Page 1  │                          │
│  │ (R/W)   │           │ (R/W)   │                          │
│  └────┬────┘           └────┬────┘                          │
│       │                     │                               │
│       ▼                     ▼                               │
│  ┌─────────┐           ┌─────────┐                          │
│  │ Frame A │           │ Frame B │  ← Copy created          │
│  │(original)│           │(modified)│                         │
│  └─────────┘           └─────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Advantages:
- fork() is fast (copy deferred)
- Memory savings (stay shared if read-only)
- Efficient when exec() discards all pages
```

### 8.2 Memory Mapped Files (mmap)

```
Principle: Map file directly to virtual address space

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Process Virtual Address Space         Disk File            │
│                                                             │
│  0x40000000 ┌────────────┐     ┌────────────┐              │
│             │            │     │            │              │
│             │  Mapped    │◀═══▶│   File     │              │
│             │  Region    │     │  Content   │              │
│             │            │     │            │              │
│  0x40100000 └────────────┘     └────────────┘              │
│                                                             │
│  Access method:                                             │
│  - Access file content like memory (using pointers)         │
│  - Auto-load from file on page fault                        │
│  - Auto-write to file on modification (or explicit msync)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Uses:
- Large file processing
- Shared memory between processes
- Dynamic library loading
```

### 8.3 Huge Pages

```
Normal Pages vs Huge Pages:

Normal Page (4KB):
- 2MB memory = 512 pages
- 512 TLB entries needed (TLB pressure)
- Large page table overhead

Huge Page (2MB):
- 2MB memory = 1 page
- Only 1 TLB entry needed
- Increased TLB coverage

┌─────────────────────────────────────────────────────────────┐
│  TLB Coverage Comparison (64 TLB entries):                  │
│                                                             │
│  4KB pages:  64 × 4KB = 256KB                               │
│  2MB pages:  64 × 2MB = 128MB                               │
│  1GB pages:  64 × 1GB = 64GB                                │
│                                                             │
│  Big benefit for databases, virtualization, large memory    │
└─────────────────────────────────────────────────────────────┘

Using in Linux:
# System configuration
echo 1024 > /proc/sys/vm/nr_hugepages

# In program
mmap(..., MAP_HUGETLB, ...);
```

---

## 9. Practice Problems

### Basic Problems

1. What are 3 main advantages of virtual memory?

2. For a 32-bit system with 4KB pages, what is the VPN and Offset of virtual address 0x12345ABC?

3. Explain the role and necessity of TLB.

### Intermediate Problems

4. Calculate the miss count for FIFO and LRU algorithms with the following access order:
   (3 frames, access: 7,0,1,2,0,3,0,4,2,3)

5. How many memory accesses occur on TLB miss with a 2-level page table?

6. Explain the operation principle and advantages of Copy-on-Write.

### Advanced Problems

7. Why do 64-bit systems use 4-level page tables?

8. Calculate the Effective Access Time for the following scenario:
   - TLB hit time: 10ns
   - TLB hit rate: 98%
   - Page table walk: 200ns (4 memory accesses)
   - Memory access: 100ns
   - Page fault rate: 0.001%
   - Page fault handling: 10ms

9. Explain how the Working Set model prevents thrashing.

<details>
<summary>Answers</summary>

1. Virtual Memory Advantages:
   - Run programs larger than physical memory
   - Memory protection (process isolation)
   - Memory sharing (libraries, etc.)
   - Fragmentation solved
   - Relocation transparency

2. Address decomposition (0x12345ABC):
   - VPN = 0x12345 (upper 20 bits)
   - Offset = 0xABC (lower 12 bits)

3. TLB Role:
   - Caches virtual-to-physical address translation results
   - Reduces page table access overhead
   - Necessity: 4-level table = 4 additional memory accesses reduced to 1 cycle

4. Miss count:
   FIFO: 7,0,1,2,0,3,0,4,2,3
   Actual calculation:
   [7,-,-] → [7,0,-] → [7,0,1] → [2,0,1] → Hit → [2,3,1] → [2,3,0] → [4,3,0] → [4,2,0] → [4,2,3]
   FIFO: 9 Miss, 1 Hit

   LRU:
   [7,-,-] → [7,0,-] → [7,0,1] → [2,0,1] → Hit(0) → [2,0,3] → Hit(0) → [4,0,3] → [4,0,2] → [4,3,2]
   LRU: 8 Miss, 2 Hit

5. 2-level page table TLB miss:
   - Page Directory access: 1
   - Page Table access: 1
   - Actual data access: 1
   - Total 3 memory accesses

6. Copy-on-Write:
   - Principle: On fork(), only copy page tables, share actual pages; copy on write
   - Advantages: fork() is fast, memory savings, efficient for exec()

7. 4-level page table reason:
   - 64-bit address space is too large
   - Single level: 2^52 / 4KB × 8B = several PB table
   - Multi-level: Only allocate tables for used regions, saving memory

8. EAT Calculation:
   TLB hit: 10ns + 100ns = 110ns
   TLB miss: 10ns + 200ns + 100ns = 310ns
   Page fault: 10ms = 10,000,000ns

   EAT = 0.98 × 0.99999 × 110 + 0.02 × 0.99999 × 310 + 0.00001 × 10,000,000
       = 107.78 + 6.20 + 100
       ≈ 214ns

9. Working Set Model:
   - Tracks Working Set size of each process
   - If physical memory < sum of all process Working Sets
   - Swap out some processes to give others enough memory
   - Each process guaranteed frames for its Working Set, preventing thrashing

</details>

---

## Next Steps

- [17_IO_Systems.md](./17_IO_Systems.md) - I/O systems, interrupts, DMA

---

## References

- Operating System Concepts (Silberschatz et al.)
- Computer Architecture: A Quantitative Approach (Hennessy & Patterson)
- [Intel Software Developer's Manual - Paging](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Linux Kernel Documentation - Memory Management](https://www.kernel.org/doc/html/latest/mm/)
