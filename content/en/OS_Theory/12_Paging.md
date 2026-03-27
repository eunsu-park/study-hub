# Paging ⭐⭐⭐

**Previous**: [Contiguous Memory Allocation](./11_Contiguous_Memory_Allocation.md) | **Next**: [Segmentation](./13_Segmentation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how paging eliminates external fragmentation
2. Perform manual page-to-frame address translation
3. Describe the structure and purpose of a page table entry
4. Explain why TLBs are critical for paging performance
5. Compare single-level, multi-level, hashed, and inverted page tables
6. Calculate the memory overhead of page tables

---

Paging is one of the most important ideas in operating systems. It lets the OS give each process the illusion of a large, contiguous address space while actually scattering the process across physical memory. This decoupling of logical and physical addresses enables memory protection, sharing, and virtual memory.

## Table of Contents

1. [Basic Concepts of Paging](#1-basic-concepts-of-paging)
2. [Address Translation](#2-address-translation)
3. [Page Table](#3-page-table)
4. [TLB (Translation Lookaside Buffer)](#4-tlb-translation-lookaside-buffer)
5. [Multi-level Page Tables](#5-multi-level-page-tables)
6. [Hashed Page Tables](#6-hashed-page-tables)
7. [Inverted Page Tables](#7-inverted-page-tables)
8. [Practice Problems](#practice-problems)

---

## 1. Basic Concepts of Paging

> Think of a bookshelf (physical memory) with numbered slots (frames) and a textbook (process) with numbered pages. The pages do not need to be placed in consecutive slots -- page 1 might go in slot 5, page 2 in slot 12. A page table is the index card that maps each page number to its slot location. This simple idea -- non-contiguous allocation with a lookup table -- eliminates external fragmentation entirely.

### 1.1 Pages and Frames

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Basic Structure of Paging                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Logical Address Space (Process)       Physical Address Space (RAM)    │
│                                                                          │
│   ┌────────────────┐                     ┌────────────────┐             │
│   │   Page 0       │ ──────────────────▶ │   Frame 1      │             │
│   ├────────────────┤                     ├────────────────┤             │
│   │   Page 1       │ ─────┐              │   Frame 2      │             │
│   ├────────────────┤      │              ├────────────────┤             │
│   │   Page 2       │ ───┐ │              │   Frame 3      │ ◀────────┐  │
│   ├────────────────┤    │ │              ├────────────────┤          │  │
│   │   Page 3       │ ─┐ │ └────────────▶ │   Frame 4      │          │  │
│   └────────────────┘  │ │                ├────────────────┤          │  │
│                       │ │                │   Frame 5      │ ◀─────┐  │  │
│                       │ │                ├────────────────┤       │  │  │
│                       │ └───────────────▶│   Frame 6      │       │  │  │
│                       │                  ├────────────────┤       │  │  │
│                       └─────────────────▶│   Frame 7      │       │  │  │
│                                          └────────────────┘       │  │  │
│                                                                   │  │  │
│   Page size = Frame size (e.g., 4KB)                             │  │  │
│   Pages can be placed in any frame!                              │  │  │
│   No external fragmentation!                                     │  │  │
│                                                                   │  │  │
│   Other process pages ─────────────────────────────────────────────┘  │  │
│   (Sharing the same physical memory)                                  │  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Terms

| Term | Description |
|------|-------------|
| **Page** | Fixed-size block of logical address space |
| **Frame** | Fixed-size block of physical memory |
| **Page Table** | Page → Frame mapping information |
| **Page Number (p)** | Identifies page in logical address |
| **Offset (d)** | Position within page/frame |

---

## 2. Address Translation

### 2.1 Logical Address Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Logical Address Structure (32-bit example)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         32-bit Logical Address                           │
│   ┌─────────────────────────────┬─────────────────┐                     │
│   │      Page Number (p)         │   Offset (d)    │                     │
│   │         20 bits              │     12 bits     │                     │
│   └─────────────────────────────┴─────────────────┘                     │
│                                                                          │
│   Page size = 2^12 = 4KB                                                │
│   Maximum pages = 2^20 = ~1 million                                     │
│   Maximum address space = 2^32 = 4GB                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Address Translation Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Address Translation Process                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Logical address: 0x00005A34                                            │
│                                                                          │
│   1. Separate address (Page size: 4KB = 0x1000)                         │
│      ┌───────────────────────────────────────┐                          │
│      │  0x00005A34                           │                          │
│      │  = 0x00005 (page number) + 0xA34 (offset)│                       │
│      └───────────────────────────────────────┘                          │
│                                                                          │
│   2. Look up page table                                                  │
│      ┌────────────┬────────────┐                                        │
│      │ Page Number│ Frame Number│                                        │
│      ├────────────┼────────────┤                                        │
│      │     0      │     2      │                                        │
│      │     1      │     7      │                                        │
│      │     2      │     1      │                                        │
│      │     3      │     5      │                                        │
│      │     4      │     8      │                                        │
│      │     5      │     3      │ ◀── Page 5 → Frame 3                  │
│      └────────────┴────────────┘                                        │
│                                                                          │
│   3. Calculate physical address                                          │
│      Physical address = (frame number × page size) + offset              │
│                       = (3 × 0x1000) + 0xA34                            │
│                       = 0x3000 + 0xA34                                  │
│                       = 0x3A34                                           │
│                                                                          │
│   Result: Logical address 0x5A34 → Physical address 0x3A34             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 C Language Implementation

```c
#include <stdio.h>
#include <stdint.h>

#define PAGE_SIZE       4096        // 4KB
#define PAGE_BITS       12          // log2(4096)
#define NUM_PAGES       1024        // Number of page table entries

typedef struct {
    uint32_t frame_number;
    uint8_t  valid;                 // Valid bit
    uint8_t  read_only;             // Read-only
    uint8_t  modified;              // Modified (dirty bit)
} PageTableEntry;

PageTableEntry page_table[NUM_PAGES];

// Logical address → Physical address translation
uint32_t translate_address(uint32_t logical_address) {
    // 1. Separate page number and offset
    uint32_t page_number = logical_address >> PAGE_BITS;
    uint32_t offset = logical_address & (PAGE_SIZE - 1);

    printf("Logical address: 0x%08X\n", logical_address);
    printf("Page number: %u, Offset: 0x%X\n", page_number, offset);

    // 2. Check page table bounds
    if (page_number >= NUM_PAGES) {
        printf("ERROR: Page number out of bounds!\n");
        return -1;
    }

    // 3. Check valid bit
    if (!page_table[page_number].valid) {
        printf("PAGE FAULT: Page %u is not in memory\n", page_number);
        return -1;  // Page fault handling required
    }

    // 4. Calculate physical address
    uint32_t frame_number = page_table[page_number].frame_number;
    uint32_t physical_address = (frame_number << PAGE_BITS) | offset;

    printf("Frame number: %u\n", frame_number);
    printf("Physical address: 0x%08X\n", physical_address);

    return physical_address;
}

int main() {
    // Initialize page table
    page_table[0] = (PageTableEntry){.frame_number = 2, .valid = 1};
    page_table[1] = (PageTableEntry){.frame_number = 7, .valid = 1};
    page_table[2] = (PageTableEntry){.frame_number = 1, .valid = 1};
    page_table[5] = (PageTableEntry){.frame_number = 3, .valid = 1};

    // Test address translation
    translate_address(0x00000A34);  // Page 0
    printf("\n");
    translate_address(0x00005A34);  // Page 5

    return 0;
}
```

### 2.4 Calculation Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Address Translation Example                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Problem: Page size = 1KB, Logical address = 4500                       │
│                                                                          │
│  Solution:                                                               │
│  1. Page size = 1024 = 2^10 bytes                                       │
│     → Offset = 10 bits                                                  │
│                                                                          │
│  2. Separate logical address:                                            │
│     Page number = 4500 / 1024 = 4 (integer division)                    │
│     Offset     = 4500 % 1024 = 404                                      │
│                                                                          │
│     Verification: 4 × 1024 + 404 = 4096 + 404 = 4500 ✓                 │
│                                                                          │
│  3. Look up page table: Page 4 → Frame 7                                │
│                                                                          │
│  4. Physical address = Frame 7 × 1024 + 404                             │
│                      = 7168 + 404                                        │
│                      = 7572                                              │
│                                                                          │
│  Binary verification:                                                    │
│  - Logical address 4500 = 0001 0001 1001 0100                           │
│                           ├─────────┤├─────────┤                        │
│                           Page 4    Offset 404                           │
│                                                                          │
│  - Physical address 7572 = 0001 1101 1001 0100                          │
│                            ├─────────┤├─────────┤                        │
│                            Frame 7   Offset 404 (same!)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Page Table

### 3.1 Page Table Entry (PTE)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Page Table Entry Structure (32-bit)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   31                               12 11         4 3 2 1 0              │
│   ┌──────────────────────────────────┬──────────┬─┬─┬─┬─┐              │
│   │        Frame Number (20 bits)    │ Reserved │D│A│U│V│              │
│   └──────────────────────────────────┴──────────┴─┴─┴─┴─┘              │
│                                                                          │
│   V (Valid): Valid bit                                                  │
│       0 = Page not in memory (page fault occurs)                        │
│       1 = Page is in memory                                             │
│                                                                          │
│   U (User): User access allowed                                         │
│       0 = Kernel mode only                                              │
│       1 = User mode access allowed                                      │
│                                                                          │
│   A (Accessed): Accessed bit                                            │
│       Hardware sets to 1 on access                                      │
│       Used by page replacement algorithms                               │
│                                                                          │
│   D (Dirty): Modified bit                                               │
│       Hardware sets to 1 on write                                       │
│       Determines if page needs to be written to disk on eviction        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Page Table Storage Location

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Page Table and PTBR                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   CPU                                                                    │
│   ┌────────────────────────────────────────┐                            │
│   │                                        │                            │
│   │   PTBR (Page Table Base Register)      │                            │
│   │   ┌────────────────────────────────┐   │                            │
│   │   │         0x00100000             │───┼───────┐                    │
│   │   └────────────────────────────────┘   │       │                    │
│   │                                        │       │                    │
│   │   PTLR (Page Table Length Register)    │       │                    │
│   │   ┌────────────────────────────────┐   │       │                    │
│   │   │            1024                │   │       │                    │
│   │   └────────────────────────────────┘   │       │                    │
│   │                                        │       │                    │
│   └────────────────────────────────────────┘       │                    │
│                                                     │                    │
│                                                     ▼                    │
│   Physical Memory                                                        │
│   ┌────────────────────────────────────────────────────┐                │
│   │                                                    │                │
│   │   Address 0x00100000                               │                │
│   │   ┌───────────────────────────────────────────┐   │                │
│   │   │    Page Table (Process A)                 │   │                │
│   │   │   ┌──────────────┬────────────────────┐   │   │                │
│   │   │   │ Page 0       │ Frame 5 | V=1     │   │   │                │
│   │   │   │ Page 1       │ Frame 2 | V=1     │   │   │                │
│   │   │   │ Page 2       │ Frame 8 | V=1     │   │   │                │
│   │   │   │ ...          │ ...                │   │   │                │
│   │   │   └──────────────┴────────────────────┘   │   │                │
│   │   └───────────────────────────────────────────┘   │                │
│   │                                                    │                │
│   └────────────────────────────────────────────────────┘                │
│                                                                          │
│   On context switch: Change PTBR value to new process's page table addr │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Memory Access Problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Two Memory Access Problem                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Memory access when using paging:                                      │
│                                                                          │
│   CPU ──▶ Generate logical address                                      │
│           │                                                              │
│           ▼                                                              │
│   [1] Access page table (Memory access #1)                              │
│           │                                                              │
│           ▼                                                              │
│   Get frame number                                                       │
│           │                                                              │
│           ▼                                                              │
│   [2] Access actual data (Memory access #2)                             │
│           │                                                              │
│           ▼                                                              │
│   Get data                                                               │
│                                                                          │
│   Problem: Memory access becomes 2x slower!                              │
│   Solution: Use TLB (Translation Lookaside Buffer)                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. TLB (Translation Lookaside Buffer)

### 4.1 TLB Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TLB Structure                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   High-speed cache inside CPU or MMU                                    │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────┐        │
│   │                          TLB                                │        │
│   ├────────────────────────────────────────────────────────────┤        │
│   │   Page Number    │    Frame Number    │    ASID    │  Valid  │        │
│   ├──────────────────┼───────────────────┼────────────┼─────────┤        │
│   │       5          │         3         │     1      │    1    │        │
│   │       2          │         7         │     1      │    1    │        │
│   │      12          │         9         │     2      │    1    │        │
│   │       0          │         1         │     1      │    1    │        │
│   │      ...         │        ...        │    ...     │   ...   │        │
│   └────────────────────────────────────────────────────────────┘        │
│                                                                          │
│   ASID (Address Space ID): Process identifier                           │
│   → No need to flush entire TLB on context switch                       │
│                                                                          │
│   Entry count: Usually 64 ~ 1024 entries (very small, but very fast)    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Address Translation Using TLB

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Address Translation Using TLB                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Logical address                                                        │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────────────────────┐                                   │
│   │      Extract page number        │                                   │
│   └─────────────────────────────────┘                                   │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────────────────────┐                                   │
│   │       Search in TLB             │                                   │
│   └─────────────────────────────────┘                                   │
│       │                                                                  │
│       ├───────────────────────────────────┐                             │
│       │                                   │                             │
│   TLB Hit                             TLB Miss                          │
│       │                                   │                             │
│       ▼                                   ▼                             │
│   Frame number                     Access page table                    │
│   immediately obtained             (memory access)                      │
│       │                                   │                             │
│       │                                   ▼                             │
│       │                           Update TLB                            │
│       │                                   │                             │
│       └───────────────────────────────────┘                             │
│                       │                                                  │
│                       ▼                                                  │
│               Calculate physical address                                 │
│                       │                                                  │
│                       ▼                                                  │
│                  Memory access                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Effective Access Time (EAT)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Effective Access Time Calculation                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Parameters:                                                            │
│   - TLB search time: ε (e.g., 10ns)                                     │
│   - Memory access time: m (e.g., 100ns)                                 │
│   - TLB hit ratio: α (e.g., 0.98 = 98%)                                │
│                                                                          │
│   EAT formula:                                                           │
│   EAT = α × (ε + m) + (1-α) × (ε + 2m)                                 │
│         ↑ TLB Hit       ↑ TLB Miss                                      │
│                                                                          │
│   Example calculation:                                                   │
│   ε = 10ns, m = 100ns, α = 0.98                                        │
│                                                                          │
│   EAT = 0.98 × (10 + 100) + 0.02 × (10 + 200)                          │
│       = 0.98 × 110 + 0.02 × 210                                        │
│       = 107.8 + 4.2                                                     │
│       = 112ns                                                            │
│                                                                          │
│   Comparison:                                                            │
│   - Without TLB: 200ns (2 memory accesses)                              │
│   - With TLB: 112ns                                                      │
│   - Performance improvement: ~44%                                        │
│                                                                          │
│   Higher TLB hit ratio results in greater performance improvement        │
│   α = 0.99: EAT = 0.99×110 + 0.01×210 = 112ns                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Multi-level Page Tables

### 5.1 Problem

For 32-bit address space with 4KB page size:
- Number of pages = 2^32 / 2^12 = 2^20 = ~1 million
- PTE size = 4 bytes
- Page table size = 1 million × 4 = 4MB (per process!)

### 5.2 Two-Level Page Table

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Two-Level Page Table                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   32-bit logical address:                                                │
│   ┌──────────────┬──────────────┬──────────────┐                        │
│   │   p1 (10bits) │   p2 (10bits) │   d (12bits)  │                        │
│   │  Outer page  │  Inner page  │   Offset     │                        │
│   └──────────────┴──────────────┴──────────────┘                        │
│          │              │              │                                 │
│          │              │              │                                 │
│          ▼              │              │                                 │
│   ┌────────────┐        │              │                                 │
│   │ Outer Page │        │              │                                 │
│   │   Table    │        │              │                                 │
│   │ (Level 1)  │        │              │                                 │
│   ├────────────┤        │              │                                 │
│   │   ...      │        │              │                                 │
│   │  p1 entry  │────────┼──────┐       │                                 │
│   │   ...      │        │      │       │                                 │
│   └────────────┘        │      │       │                                 │
│                         │      ▼       │                                 │
│                         │  ┌────────────┐                                │
│                         │  │ Inner Page │                                │
│                         │  │   Table    │                                │
│                         │  │ (Level 2)  │                                │
│                         │  ├────────────┤                                │
│                         │  │   ...      │                                │
│                         └▶ │  p2 entry  │───▶ Frame number              │
│                            │   ...      │         │                      │
│                            └────────────┘         │                      │
│                                                   │                      │
│                                                   ▼                      │
│                                    Physical addr = Frame × 4KB + d       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Four-Level Page Table for 64-bit Systems

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 64-bit 4-Level Page Table (x86-64)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   64-bit virtual address (actual usage: 48 bits)                        │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │ Unused │  PML4  │  PDPT  │   PD   │   PT   │    Offset    │     │
│   │ (16)   │  (9)   │  (9)   │  (9)   │  (9)   │     (12)     │     │
│   └──────────────────────────────────────────────────────────────┘     │
│               │        │        │        │            │                 │
│               ▼        ▼        ▼        ▼            │                 │
│           ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐    │                 │
│   CR3 ──▶ │ PML4  │─▶│ PDPT │─▶│  PD   │─▶│  PT   │───┼───▶ Physical addr│
│           │ Table │  │ Table│  │ Table │  │ Table │   │                 │
│           └───────┘  └───────┘  └───────┘  └───────┘   │                 │
│                                                         │                 │
│   Each level: 512 entries (2^9), 8 bytes/entry         │                 │
│   Table size: 512 × 8 = 4KB (= 1 page)                 │                 │
│                                                         │                 │
│   PML4: Page Map Level 4                               │                 │
│   PDPT: Page Directory Pointer Table                   │                 │
│   PD: Page Directory                                   │                 │
│   PT: Page Table                                       │                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Advantages of Multi-level Tables

```c
// Memory savings with multi-level page tables

// Process memory usage:
// - Code: 0x00000000 ~ 0x00100000 (1MB, 256 pages)
// - Stack: 0xBFF00000 ~ 0xC0000000 (1MB, 256 pages)

// Single page table: 4MB needed (all 1 million entries)

// Two-level page table:
// - Level 1 table: 4KB (1024 entries)
// - Level 2 tables: Only what's needed!
//   - Code region: 1 × 4KB
//   - Stack region: 1 × 4KB
// Total: 4KB + 4KB + 4KB = 12KB (saved from 4MB to 12KB!)

typedef struct {
    uint32_t present : 1;
    uint32_t writable : 1;
    uint32_t user : 1;
    uint32_t reserved : 9;
    uint32_t frame : 20;       // Next level table or frame
} PageTableEntry;

// Two-level address translation
uint32_t translate_two_level(uint32_t virtual_addr) {
    uint32_t p1 = (virtual_addr >> 22) & 0x3FF;   // Upper 10 bits
    uint32_t p2 = (virtual_addr >> 12) & 0x3FF;   // Middle 10 bits
    uint32_t offset = virtual_addr & 0xFFF;        // Lower 12 bits

    // Access level 1 table
    PageTableEntry* level1 = (PageTableEntry*)PTBR;
    if (!level1[p1].present) {
        // Level 2 table doesn't exist
        page_fault_handler();
        return -1;
    }

    // Access level 2 table
    PageTableEntry* level2 = (PageTableEntry*)(level1[p1].frame << 12);
    if (!level2[p2].present) {
        // Page not in memory
        page_fault_handler();
        return -1;
    }

    // Calculate physical address
    uint32_t frame = level2[p2].frame;
    return (frame << 12) | offset;
}
```

---

## 6. Hashed Page Tables

### 6.1 Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Hashed Page Table                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Virtual page number                                                    │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────┐                                                       │
│   │  Hash func  │                                                       │
│   └─────────────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│   Hash table                                                             │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │ Index │                    Chain                              │     │
│   ├────────┼───────────────────────────────────────────────────────┤     │
│   │   0    │ [p=102, f=5] → [p=2050, f=12] → NULL                 │     │
│   │   1    │ NULL                                                  │     │
│   │   2    │ [p=55, f=8] → NULL                                   │     │
│   │   3    │ [p=1003, f=3] → [p=7, f=22] → [p=4099, f=1] → NULL  │     │
│   │  ...   │ ...                                                   │     │
│   └────────┴───────────────────────────────────────────────────────┘     │
│                                                                          │
│   p = Virtual page number                                               │
│   f = Physical frame number                                             │
│                                                                          │
│   Advantage: Efficient for sparse address spaces (64-bit systems)       │
│   Disadvantage: Chain search required on hash collision                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Implementation

```c
#define HASH_TABLE_SIZE 1024

typedef struct HashEntry {
    uint64_t virtual_page;
    uint64_t physical_frame;
    struct HashEntry* next;
} HashEntry;

HashEntry* hash_table[HASH_TABLE_SIZE];

// Hash function
uint32_t hash(uint64_t virtual_page) {
    return virtual_page % HASH_TABLE_SIZE;
}

// Search for frame number
int64_t lookup(uint64_t virtual_page) {
    uint32_t index = hash(virtual_page);
    HashEntry* entry = hash_table[index];

    while (entry != NULL) {
        if (entry->virtual_page == virtual_page) {
            return entry->physical_frame;
        }
        entry = entry->next;
    }

    return -1;  // Page fault
}

// Add mapping
void insert(uint64_t virtual_page, uint64_t physical_frame) {
    uint32_t index = hash(virtual_page);

    HashEntry* new_entry = malloc(sizeof(HashEntry));
    new_entry->virtual_page = virtual_page;
    new_entry->physical_frame = physical_frame;
    new_entry->next = hash_table[index];

    hash_table[index] = new_entry;
}
```

---

## 7. Inverted Page Tables

### 7.1 Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Inverted Page Table                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional page table:                                               │
│   - Separate table per process                                          │
│   - Virtual address → Physical address mapping                          │
│   - Table size proportional to virtual address space                    │
│                                                                          │
│   Inverted page table:                                                  │
│   - One table for entire system                                         │
│   - Entry per physical frame                                            │
│   - Table size proportional to physical memory size                     │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │ Frame Number │    PID    │  Virtual Page Number  │   Protection │     │
│   ├─────────────┼───────────┼───────────────────┼───────────────┤     │
│   │      0      │     1     │       0x1234      │     R/W       │     │
│   │      1      │     2     │       0x0001      │     R/W       │     │
│   │      2      │     1     │       0x1235      │     R/O       │     │
│   │      3      │     3     │       0x5678      │     R/W       │     │
│   │     ...     │    ...    │        ...        │      ...      │     │
│   │      N      │     2     │       0x0002      │     R/W       │     │
│   └─────────────┴───────────┴───────────────────┴───────────────┘     │
│                                                                          │
│   Search: (PID, virtual page number) to find frame number               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Advantages and Disadvantages

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Inverted Page Table Pros/Cons                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Advantages:                                                            │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ 1. Memory savings                                                │   │
│   │    - Physical memory 4GB, frame 4KB                             │   │
│   │    - Entry count = 4GB / 4KB = 1M entries                       │   │
│   │    - 16 bytes per entry → Total 16MB (independent of # processes!)│   │
│   │                                                                  │   │
│   │ 2. Efficient for 64-bit systems                                 │   │
│   │    - Virtual address space is very large but                    │   │
│   │    - Physical memory is relatively small                        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Disadvantages:                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ 1. Search time                                                   │   │
│   │    - Sequential search: O(n) - too slow                         │   │
│   │    - Hash table required                                        │   │
│   │                                                                  │   │
│   │ 2. Difficult to share memory                                    │   │
│   │    - Only one (PID, page) pair per frame                       │   │
│   │    - Shared memory implementation complex                       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Usage examples: IBM PowerPC, UltraSPARC, IA-64                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

### Problem 1: Address Translation
Calculate the physical address given the following conditions with a page size of 4KB:

- Logical address: 25000
- Page table: Page 6 → Frame 4

<details>
<summary>Show Answer</summary>

```
1. Page size = 4KB = 4096 bytes

2. Page number = 25000 / 4096 = 6 (integer division)
   Offset = 25000 % 4096 = 424

   Verification: 6 × 4096 + 424 = 24576 + 424 = 25000 ✓

3. Look up page table: Page 6 → Frame 4

4. Physical address = Frame 4 × 4096 + 424
                    = 16384 + 424
                    = 16808
```

</details>

### Problem 2: TLB EAT Calculation
Calculate the Effective Access Time (EAT) given:

- TLB access time: 20ns
- Memory access time: 100ns
- TLB hit ratio: 95%

<details>
<summary>Show Answer</summary>

```
EAT = α × (ε + m) + (1-α) × (ε + 2m)
    = 0.95 × (20 + 100) + 0.05 × (20 + 200)
    = 0.95 × 120 + 0.05 × 220
    = 114 + 11
    = 125ns

Comparison: Without TLB 200ns, With TLB 125ns (37.5% improvement)
```

</details>

### Problem 3: Page Table Size
For a system with 32-bit virtual address space, 4KB pages, and 4-byte PTEs:
1. What is the size of a single page table?
2. How would address bits be allocated for a two-level page table?

<details>
<summary>Show Answer</summary>

```
1. Single page table:
   - Number of pages = 2^32 / 2^12 = 2^20 (~1 million)
   - PTE size = 4 bytes
   - Table size = 2^20 × 4 = 4MB

2. Two-level page table:
   - Total 32 bits = offset(12 bits) + page table(20 bits)
   - Split 20 bits into two levels: 10 bits + 10 bits

   Address structure: [p1: 10 bits][p2: 10 bits][offset: 12 bits]

   Each level table:
   - Entry count = 2^10 = 1024
   - Table size = 1024 × 4 = 4KB (fits exactly in 1 page)
```

</details>

### Problem 4: Multi-level Table Memory
Calculate the total size of a two-level page table when a process uses only these regions:

- Code: 0x00000000 ~ 0x00400000 (4MB)
- Data: 0x10000000 ~ 0x10100000 (1MB)
- Stack: 0xBFF00000 ~ 0xC0000000 (1MB)

<details>
<summary>Show Answer</summary>

```
Page size = 4KB, two-level table (10 bits + 10 bits + 12 bits)

Level 1 index analysis:
- Code (0x00000000~0x00400000): p1 = 0
- Data (0x10000000~0x10100000): p1 = 64
- Stack (0xBFF00000~0xC0000000): p1 = 767

Required tables:
- Level 1 table: 1 × 4KB = 4KB
- Level 2 tables: 3 × 4KB = 12KB (for indices 0, 64, 767)

Total size = 4KB + 12KB = 16KB

Comparison: Would need 4MB for single table
Savings: (4MB - 16KB) / 4MB = 99.6% savings!
```

</details>

### Problem 5: Inverted Page Table Design
Design an inverted page table for a system with 1GB physical memory and 8KB page size.
- Fields and sizes required in entry
- Total size of table

<details>
<summary>Show Answer</summary>

```
1. Frame count = 1GB / 8KB = 2^30 / 2^13 = 2^17 = 131,072 frames

2. Entry structure:
   - PID: 16 bits (max 65536 processes)
   - Virtual page number: 51 bits (64-bit addr - 13-bit offset)
   - Protection bits: 4 bits (R/W/X/Valid)
   - Other: 9 bits (spare)
   Total: 80 bits = 10 bytes (or pad to 16 bytes)

3. Table size:
   - Entry count: 131,072
   - Entry size: 16 bytes
   - Total size: 131,072 × 16 = 2MB

   This is 0.2% of physical memory (1GB)
```

</details>

---

## Next Steps

Let's learn about segment-based memory management in [13_Segmentation.md](./13_Segmentation.md)!

---

## References

- Silberschatz, "Operating System Concepts" Chapter 9
- Intel 64 and IA-32 Architectures Software Developer's Manual
- AMD64 Architecture Programmer's Manual
- Linux kernel source: `arch/x86/mm/pgtable.c`
