# Segmentation ⭐⭐⭐

**Previous**: [Paging](./12_Paging.md) | **Next**: [Virtual Memory](./14_Virtual_Memory.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain segmentation as a logical view of memory that reflects how programmers think about programs
2. Describe the structure of a segment table entry, including base, limit, and protection fields
3. Perform segment-based address translation given a segment number and offset
4. Compare segmentation with paging across multiple dimensions including fragmentation, protection, and sharing
5. Explain segmented paging and why modern systems combine both approaches
6. Analyze how Intel x86 architecture implements segmentation in protected mode
7. Distinguish between external fragmentation in segmentation and internal fragmentation in paging

---

While paging divides memory into fixed-size blocks for the OS's convenience, segmentation divides it according to the program's logical structure -- code, data, stack, heap. Segmentation reflects how programmers actually think about memory, making it a natural complement to paging. Understanding segmentation is essential because modern systems like x86 combine both techniques, and concepts like segment-level protection underpin memory safety in every OS you use today.

> **Analogy**: Segmentation mirrors the programmer's mental model: code, data, stack, and heap are separate logical units, each with its own base address and size limit. Think of it like a filing cabinet where each drawer (segment) holds a different category of documents -- one for contracts, one for invoices, one for correspondence. Each drawer can be a different size, locked independently, and shared with specific people. Paging, by contrast, is like cutting all documents into same-sized pages regardless of content.

## Table of Contents

1. [Segment Concept](#1-segment-concept)
2. [Segment Table](#2-segment-table)
3. [Address Translation](#3-address-translation)
4. [Protection and Sharing](#4-protection-and-sharing)
5. [Paging vs Segmentation Comparison](#5-paging-vs-segmentation-comparison)
6. [Combining Segmentation + Paging](#6-combining-segmentation--paging)
7. [Intel x86 Segmentation](#7-intel-x86-segmentation)
8. [Practice Problems](#practice-problems)

---

## 1. Segment Concept

### 1.1 Programmer's View of Memory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Memory from Programmer's View                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Programmers perceive memory not as a contiguous byte array,           │
│   but as a collection of logical units.                                 │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────┐       │
│   │                        Program                               │       │
│   ├─────────────────────────────────────────────────────────────┤       │
│   │                                                              │       │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │       │
│   │   │    main()    │  │  Function A()│  │  Function B()│      │       │
│   │   │     code     │  │     code     │  │     code     │      │       │
│   │   └──────────────┘  └──────────────┘  └──────────────┘      │       │
│   │                                                              │       │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │       │
│   │   │ Global vars  │  │  Constants   │  │    Stack     │      │       │
│   │   │   (data)     │  │  (read-only) │  │ (local vars) │      │       │
│   │   └──────────────┘  └──────────────┘  └──────────────┘      │       │
│   │                                                              │       │
│   │   ┌──────────────┐  ┌──────────────┐                        │       │
│   │   │ Symbol table │  │    Heap      │                        │       │
│   │   │  (debug)     │  │ (dynamic)    │                        │       │
│   │   └──────────────┘  └──────────────┘                        │       │
│   │                                                              │       │
│   └─────────────────────────────────────────────────────────────┘       │
│                                                                          │
│   Each logical unit = segment                                           │
│   Each segment has different size, different protection attributes      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Segment Characteristics

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Segment Characteristics                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. Variable Size                                                      │
│      - Unlike pages, not fixed size                                     │
│      - Size varies according to logical unit                            │
│                                                                          │
│   2. Logical Separation                                                 │
│      ┌──────────────┐                                                   │
│      │ Code Segment │ → Executable, read-only                           │
│      ├──────────────┤                                                   │
│      │ Data Segment │ → Read/write enabled                              │
│      ├──────────────┤                                                   │
│      │ Stack Segment│ → Auto-expanding, grows downward                  │
│      ├──────────────┤                                                   │
│      │ Heap Segment │ → Dynamic allocation, grows upward                │
│      └──────────────┘                                                   │
│                                                                          │
│   3. Easy Protection                                                    │
│      - Different access rights for each segment                         │
│      - Prevent code modification, prevent stack execution, etc.         │
│                                                                          │
│   4. Easy Sharing                                                       │
│      - Code segment shared by multiple processes                        │
│      - Suitable for shared library implementation                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Segment Table

### 2.1 Segment Table Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Segment Table                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Segment Table Base Register (STBR): 0x5000                           │
│   Segment Table Length Register (STLR): 5                              │
│                                                                          │
│   ┌────────────┬────────────────┬────────────┬───────────────────────┐  │
│   │ Segment    │ Base Address   │   Limit    │       Protection      │  │
│   │  Number    │ (Base)         │  (Limit)   │                       │  │
│   ├────────────┼────────────────┼────────────┼───────────────────────┤  │
│   │     0      │   0x00000      │   1400     │   R-X (code)         │  │
│   │     1      │   0x06300      │   400      │   R-- (const)        │  │
│   │     2      │   0x04300      │   1100     │   RW- (data)         │  │
│   │     3      │   0x03200      │   1000     │   RW- (stack)        │  │
│   │     4      │   0x04700      │   2000     │   RW- (heap)         │  │
│   └────────────┴────────────────┴────────────┴───────────────────────┘  │
│                                                                          │
│   Base: Physical memory start address of segment                        │
│   Limit: Maximum size of segment (bytes)                                │
│   Protection: R(read), W(write), X(execute)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Physical Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Segments in Physical Memory                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Physical Memory                                                       │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │ 0x00000 ┌────────────────────────────────────┐                   │  │
│   │         │        Segment 0 (code)            │                   │  │
│   │         │           1400 bytes               │                   │  │
│   │ 0x00578 └────────────────────────────────────┘                   │  │
│   │         ┌────────────────────────────────────┐                   │  │
│   │         │            Free space              │                   │  │
│   │ 0x03200 └────────────────────────────────────┘                   │  │
│   │         ┌────────────────────────────────────┐                   │  │
│   │         │       Segment 3 (stack)            │                   │  │
│   │         │           1000 bytes               │                   │  │
│   │ 0x035E8 └────────────────────────────────────┘                   │  │
│   │         ┌────────────────────────────────────┐                   │  │
│   │ 0x04300 │       Segment 2 (data)             │                   │  │
│   │         │           1100 bytes               │                   │  │
│   │ 0x0474C └────────────────────────────────────┘                   │  │
│   │         ┌────────────────────────────────────┐                   │  │
│   │ 0x04700 │        Segment 4 (heap)            │                   │  │
│   │         │           2000 bytes               │                   │  │
│   │ 0x04ED0 └────────────────────────────────────┘                   │  │
│   │         ┌────────────────────────────────────┐                   │  │
│   │ 0x06300 │       Segment 1 (const)            │                   │  │
│   │         │           400 bytes                │                   │  │
│   │ 0x06490 └────────────────────────────────────┘                   │  │
│   │         ...                                                      │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   Note: Segments are contiguous in physical memory, but not ordered     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Address Translation

### 3.1 Logical Address Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Segment Logical Address                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Logical address = <segment number, offset>                            │
│                                                                          │
│   Example: <2, 400> = 400th byte of segment 2                          │
│                                                                          │
│   ┌─────────────────┬────────────────────────────────┐                  │
│   │ Segment number(s)│         Offset(d)             │                  │
│   │     4 bits       │          12 bits              │                  │
│   └─────────────────┴────────────────────────────────┘                  │
│                                                                          │
│   In this example:                                                      │
│   - Maximum 16 segments (2^4)                                           │
│   - Maximum 4KB per segment (2^12)                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Address Translation Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Segment Address Translation                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Logical address: <2, 400>                                             │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │   1. Extract segment number: s = 2                              │   │
│   │                                                                  │   │
│   │   2. Range check: s < STLR?                                     │   │
│   │      2 < 5 → OK                                                 │   │
│   │                                                                  │   │
│   │   3. Segment table lookup                                       │   │
│   │      segment[2] = {base: 0x04300, limit: 1100}                  │   │
│   │                                                                  │   │
│   │   4. Limit check: d < limit?                                    │   │
│   │      400 < 1100 → OK                                            │   │
│   │      (If d >= limit, TRAP occurs!)                              │   │
│   │                                                                  │   │
│   │   5. Physical address calculation                               │   │
│   │      Physical address = base + d                                │   │
│   │                       = 0x04300 + 400                           │   │
│   │                       = 0x04300 + 0x190                         │   │
│   │                       = 0x04490                                 │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Result: <2, 400> → 0x04490                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Hardware Implementation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Segmentation Hardware                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         Logical address                                 │
│                    ┌───────┬────────┐                                   │
│                    │   s   │   d    │                                   │
│                    └───┬───┴────┬───┘                                   │
│                        │        │                                        │
│                        ▼        │                                        │
│   ┌─────────────────────────┐   │                                       │
│   │   Segment Table         │   │                                       │
│   │   (STBR base)           │   │                                       │
│   │                         │   │                                       │
│   │   ┌───────┬─────────┐   │   │                                       │
│   │   │ limit │  base   │◀──┘   │                                       │
│   │   └───┬───┴────┬────┘       │                                       │
│   └───────┼────────┼────────────┘                                       │
│           │        │            │                                        │
│           ▼        ▼            │                                        │
│   ┌────────────┐ ┌────────┐     │                                       │
│   │ Comparator │ │  Adder │◀────┘                                       │
│   │  d < limit │ │ base+d │                                             │
│   └─────┬──────┘ └───┬────┘                                             │
│         │            │                                                   │
│     ┌───┴───┐        │                                                  │
│     │  yes  │  no    │                                                  │
│     │   │   │   │    │                                                  │
│     ▼   │   ▼   │    ▼                                                  │
│    OK   │ TRAP  │  Physical address                                     │
│         │       │                                                        │
│         └───────┘                                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 C Language Implementation

```c
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#define MAX_SEGMENTS 16

typedef struct {
    uint32_t base;      // Base address
    uint32_t limit;     // Limit (size)
    bool valid;         // Valid bit
    uint8_t protection; // Protection bits (R=1, W=2, X=4)
} SegmentTableEntry;

SegmentTableEntry segment_table[MAX_SEGMENTS];
int stlr = 0;  // Segment Table Length Register

#define PROT_READ  1
#define PROT_WRITE 2
#define PROT_EXEC  4

// Logical address → Physical address translation
int translate_segment_address(uint16_t segment, uint32_t offset,
                               uint8_t access_type, uint32_t* physical) {
    // 1. Segment number range check
    if (segment >= stlr) {
        printf("ERROR: Invalid segment number %d (STLR=%d)\n", segment, stlr);
        return -1;  // Segmentation Fault
    }

    // 2. Validity check
    SegmentTableEntry* entry = &segment_table[segment];
    if (!entry->valid) {
        printf("ERROR: Segment %d is not valid\n", segment);
        return -1;  // Segmentation Fault
    }

    // 3. Limit check
    // Why: Hardware checks the limit BEFORE adding the offset to the base address.
    // This prevents buffer overflows into adjacent segments -- if a buggy program
    // tries to access beyond its segment boundary, the hardware traps immediately
    // rather than silently corrupting another segment's memory.
    if (offset >= entry->limit) {
        printf("ERROR: Offset %u >= Limit %u in segment %d\n",
               offset, entry->limit, segment);
        return -1;  // Segmentation Fault
    }

    // 4. Protection check
    if ((entry->protection & access_type) != access_type) {
        printf("ERROR: Protection violation in segment %d\n", segment);
        printf("  Required: 0x%x, Allowed: 0x%x\n",
               access_type, entry->protection);
        return -1;  // Protection Fault
    }

    // 5. Physical address calculation
    // Why: The base+limit pair in each segment table entry provides both relocation
    // and protection in a single mechanism. The base address relocates the segment
    // to anywhere in physical memory (so segments don't need contiguous logical-to-physical
    // mapping), while the limit ensures every access stays within bounds. This is why
    // segmentation gives per-segment protection "for free" -- the same hardware that
    // translates addresses also enforces boundaries.
    *physical = entry->base + offset;
    return 0;
}

int main() {
    // Initialize segment table
    segment_table[0] = (SegmentTableEntry){
        .base = 0x00000, .limit = 1400, .valid = true,
        .protection = PROT_READ | PROT_EXEC  // Code segment
    };
    segment_table[1] = (SegmentTableEntry){
        .base = 0x06300, .limit = 400, .valid = true,
        .protection = PROT_READ  // Constant segment
    };
    segment_table[2] = (SegmentTableEntry){
        .base = 0x04300, .limit = 1100, .valid = true,
        .protection = PROT_READ | PROT_WRITE  // Data segment
    };
    stlr = 3;

    uint32_t physical;

    // Test 1: Normal access
    if (translate_segment_address(2, 400, PROT_READ, &physical) == 0) {
        printf("Segment 2, Offset 400 -> Physical 0x%X\n", physical);
    }

    // Test 2: Limit exceeded
    translate_segment_address(2, 1200, PROT_READ, &physical);

    // Test 3: Protection violation (write attempt to code segment)
    translate_segment_address(0, 100, PROT_WRITE, &physical);

    return 0;
}
```

---

## 4. Protection and Sharing

### 4.1 Segment Protection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Segment Protection Bits                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Protection field in segment descriptor:                               │
│                                                                          │
│   ┌─────┬─────┬─────┬─────────────────────────────┐                     │
│   │  R  │  W  │  X  │         Description         │                     │
│   ├─────┼─────┼─────┼─────────────────────────────┤                     │
│   │  1  │  0  │  1  │  Code segment (exec/read)   │                     │
│   │  1  │  0  │  0  │  Const segment (read-only)  │                     │
│   │  1  │  1  │  0  │  Data segment (read/write)  │                     │
│   │  0  │  0  │  0  │  No access                  │                     │
│   └─────┴─────┴─────┴─────────────────────────────┘                     │
│                                                                          │
│   Additional protection attributes:                                     │
│   - DPL (Descriptor Privilege Level): 0-3, accessible privilege level   │
│   - Present: Whether segment is in memory                               │
│   - Accessed: Access status (used for page replacement)                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Segment Sharing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Segment Sharing                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Process A                          Process B                          │
│   ┌─────────────────┐                ┌─────────────────┐                │
│   │ Segment Table   │                │ Segment Table   │                │
│   ├─────────────────┤                ├─────────────────┤                │
│   │ 0: Code         │────────┐       │ 0: Code         │────────┐      │
│   │   base=0x1000   │        │       │   base=0x1000   │        │      │
│   │   limit=5000    │        │       │   limit=5000    │        │      │
│   ├─────────────────┤        │       ├─────────────────┤        │      │
│   │ 1: Data         │───┐    │       │ 1: Data         │───┐    │      │
│   │   base=0x8000   │   │    │       │   base=0xC000   │   │    │      │
│   │   limit=3000    │   │    │       │   limit=4000    │   │    │      │
│   └─────────────────┘   │    │       └─────────────────┘   │    │      │
│                         │    │                             │    │      │
│                         │    ▼                             │    ▼      │
│   Physical Memory       │    ┌─────────────────────────────┼────┐      │
│   ┌─────────────────────┼────│    Shared code segment      │────┤      │
│   │ 0x1000              │    │      (libc.so etc)          │    │      │
│   │                     │    │        5000 bytes           │    │      │
│   │                     │    └─────────────────────────────┼────┘      │
│   │                     │                                  │           │
│   │                     ▼                                  ▼           │
│   │ 0x8000        ┌──────────────┐                ┌──────────────┐    │
│   │               │ A's data     │  0xC000        │ B's data     │    │
│   │               │ (separate)   │                │ (separate)   │    │
│   │               └──────────────┘                └──────────────┘    │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   Code segment: Shared (reference same physical address)                │
│   Data segment: Separate (each process has its own)                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Paging vs Segmentation Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Paging vs Segmentation Comparison                      │
├───────────────────────┬─────────────────────┬───────────────────────────┤
│        Aspect         │      Paging         │      Segmentation         │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Unit                  │ Fixed size (pages)  │ Variable size (segments)  │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Programmer awareness  │ Transparent (unaware)│ Aware (logical units)    │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ External fragmentation│ None                │ Exists                    │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Internal fragmentation│ Yes (last page)     │ None                      │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Table size            │ Can be large        │ Proportional to segments  │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Memory allocation     │ Simple              │ Complex (First-Fit etc)   │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Protection unit       │ Page level          │ Logical unit (flexible)   │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Sharing               │ Possible but complex│ Easy by logical unit      │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Modern usage          │ Primarily used      │ Combined with paging      │
└───────────────────────┴─────────────────────┴───────────────────────────┘
```

---

## 6. Combining Segmentation + Paging

### 6.1 Segmentation with Paging

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Segment + Paging Combined Address Translation             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Logical address                                                       │
│   ┌────────────┬────────────┬──────────────┐                            │
│   │ Segment(s) │ Page(p)    │  Offset(d)   │                            │
│   └─────┬──────┴─────┬──────┴──────┬───────┘                            │
│         │            │             │                                     │
│         ▼            │             │                                     │
│   ┌──────────────┐   │             │                                     │
│   │ Segment      │   │             │                                     │
│   │   Table      │   │             │                                     │
│   ├──────────────┤   │             │                                     │
│   │ Segment s:   │   │             │                                     │
│   │ Page table   │───┘             │                                     │
│   │ start addr  │                  │                                     │
│   └──────┬───────┘                  │                                     │
│          │                          │                                     │
│          ▼                          │                                     │
│   ┌──────────────┐                  │                                     │
│   │ Page Table   │                  │                                     │
│   │ (per segment)│                  │                                     │
│   ├──────────────┤                  │                                     │
│   │ Page p:      │                  │                                     │
│   │ Frame number │───────────┐      │                                     │
│   └──────────────┘           │      │                                     │
│                              │      │                                     │
│                              ▼      ▼                                     │
│                        ┌──────────────┐                                   │
│                        │Physical addr │                                   │
│                        │= frame×size  │                                   │
│                        │  + offset    │                                   │
│                        └──────────────┘                                   │
│                                                                          │
│   Advantage: Segment's logical separation + Paging's no external frag   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 MULTICS System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MULTICS Memory Management                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   34-bit virtual address:                                               │
│   ┌──────────────┬──────────────┬──────────────┐                        │
│   │ Segment(18)  │ Page(6)      │  Offset(10)  │                        │
│   └──────────────┴──────────────┴──────────────┘                        │
│                                                                          │
│   - Number of segments: 2^18 = 256K segments                            │
│   - Pages per segment: 2^6 = 64 pages                                   │
│   - Page size: 2^10 = 1KB                                               │
│   - Maximum segment size: 64 × 1KB = 64KB                               │
│                                                                          │
│   Address translation:                                                  │
│   1. Lookup segment table with segment number                           │
│   2. Get page table location for that segment                           │
│   3. Lookup page table with page number                                 │
│   4. Frame number + offset = physical address                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Intel x86 Segmentation

### 7.1 Protected Mode Segment Registers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    x86 Segment Registers                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   16-bit Segment Selector                                               │
│   ┌────────────────────────────┬─────┬───────┐                          │
│   │    Index (13 bits)         │ TI  │  RPL  │                          │
│   └────────────────────────────┴─────┴───────┘                          │
│                                                                          │
│   TI: Table Indicator (0=GDT, 1=LDT)                                    │
│   RPL: Requested Privilege Level (0-3)                                  │
│                                                                          │
│   Segment registers:                                                    │
│   ┌──────┬────────────────────────────────────────┐                     │
│   │  CS  │ Code Segment - currently executing code│                     │
│   │  DS  │ Data Segment - default data            │                     │
│   │  SS  │ Stack Segment - stack                  │                     │
│   │  ES  │ Extra Segment - additional data        │                     │
│   │  FS  │ Extra segment (thread local storage)   │                     │
│   │  GS  │ Extra segment (used by kernel)         │                     │
│   └──────┴────────────────────────────────────────┘                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Segment Descriptor

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  x86 Segment Descriptor (64 bits)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Bits:  63      56 55   52 51   48 47          40 39   32              │
│         ┌─────────┬───────┬───────┬──────────────┬───────┐              │
│   Upper │Base[31:24]│Flags │Limit  │ Access Byte  │Base   │              │
│   32bits│  (8bits) │(4bits)│[19:16]│   (8bits)    │[23:16]│              │
│         └─────────┴───────┴───────┴──────────────┴───────┘              │
│                                                                          │
│   Bits:  31              16 15                   0                       │
│         ┌─────────────────┬─────────────────────┐                       │
│   Lower │  Base[15:0]     │   Limit[15:0]       │                       │
│   32bits│    (16bits)     │     (16bits)        │                       │
│         └─────────────────┴─────────────────────┘                       │
│                                                                          │
│   Base: 32-bit segment start address (stored distributed)               │
│   Limit: 20-bit segment size                                            │
│                                                                          │
│   Flags (4 bits):                                                       │
│   ┌────┬────┬────┬────┐                                                 │
│   │ G  │D/B │ L  │AVL │                                                 │
│   └────┴────┴────┴────┘                                                 │
│   G: Granularity (0=byte, 1=4KB unit)                                   │
│   D/B: Default operation size (0=16bit, 1=32bit)                        │
│   L: 64-bit code segment                                                │
│   AVL: Available for system software                                    │
│                                                                          │
│   Access Byte:                                                          │
│   ┌────┬─────┬────┬────┬────┬────┬────┬────┐                            │
│   │ P  │ DPL │ S  │ E  │DC  │RW  │ A  │    │                            │
│   └────┴─────┴────┴────┴────┴────┴────┴────┘                            │
│   P: Present bit                                                        │
│   DPL: Descriptor Privilege Level (0-3)                                 │
│   S: Descriptor type (0=system, 1=code/data)                            │
│   E: Executable (0=data, 1=code)                                        │
│   DC: Direction/Conforming                                              │
│   RW: Readable(code)/Writable(data)                                     │
│   A: Accessed                                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 x86 Address Translation (Segmentation + Paging)

```
┌─────────────────────────────────────────────────────────────────────────┐
│               x86 Complete Address Translation Process                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Logical address                                                       │
│   ┌──────────────────┬────────────────────────────────┐                 │
│   │ Segment selector │         Offset                │                 │
│   │    (16 bits)     │        (32 bits)              │                 │
│   └────────┬─────────┴───────────────┬────────────────┘                 │
│            │                         │                                   │
│            ▼                         │                                   │
│   ┌─────────────────────┐            │                                   │
│   │  Segmentation stage │            │                                   │
│   │                     │            │                                   │
│   │  GDT/LDT lookup     │            │                                   │
│   │  Base + Offset      │◀───────────┘                                   │
│   │  = Linear address   │                                                │
│   └──────────┬──────────┘                                                │
│              │                                                           │
│              ▼ Linear address                                           │
│   ┌─────────────────────┐                                                │
│   │   Paging stage      │                                                │
│   │                     │                                                │
│   │  CR3 → Page         │                                                │
│   │  directory → table  │                                                │
│   │  → Frame            │                                                │
│   └──────────┬──────────┘                                                │
│              │                                                           │
│              ▼                                                           │
│         Physical address                                                │
│                                                                          │
│   Modern OS (Linux, Windows):                                           │
│   - Simplified segmentation (Base=0, Limit=max)                         │
│   - Only paging is actually used                                        │
│   - Linear address = logical address (flat memory model)                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.4 Linux Segmentation Usage

```c
// Segment settings in Linux x86 (arch/x86/include/asm/segment.h)

// GDT entry index (Linux kernel)
#define GDT_ENTRY_KERNEL_CS     1
#define GDT_ENTRY_KERNEL_DS     2
#define GDT_ENTRY_DEFAULT_USER_CS   3
#define GDT_ENTRY_DEFAULT_USER_DS   4

// Segment selector calculation
// Index << 3 | TI(0=GDT) | RPL
#define __KERNEL_CS (GDT_ENTRY_KERNEL_CS * 8)      // 0x08
#define __KERNEL_DS (GDT_ENTRY_KERNEL_DS * 8)      // 0x10
#define __USER_CS   (GDT_ENTRY_DEFAULT_USER_CS * 8 + 3)  // 0x1B (RPL=3)
#define __USER_DS   (GDT_ENTRY_DEFAULT_USER_DS * 8 + 3)  // 0x23 (RPL=3)

// Flat memory model: All segments start at 0, up to 4GB
// Base = 0x00000000
// Limit = 0xFFFFFFFF (4GB if G=1)

/*
 * Linux uses segmentation minimally:
 * - Kernel/user privilege separation (CS, DS segments)
 * - Thread local storage (FS, GS segments)
 * - Actual memory protection is done by paging
 */
```

---

## Practice Problems

### Problem 1: Segment Address Translation
Translate logical address <1, 500> to physical address using the following segment table.

| Segment | Base | Limit |
|---------|------|-------|
| 0 | 1000 | 600 |
| 1 | 2000 | 400 |
| 2 | 3000 | 800 |

<details>
<summary>Show Answer</summary>

```
1. Segment number: 1
2. Offset: 500
3. Segment 1's Limit: 400

Since offset(500) >= Limit(400), Segmentation Fault!

Address translation failed - segment range exceeded
```

</details>

### Problem 2: Segment Protection
Determine if protection violations occur in the following scenarios.

Segment table:
| Segment | Base | Limit | Protection |
|---------|------|-------|------------|
| 0 (code) | 0x1000 | 2000 | R-X |
| 1 (data) | 0x5000 | 3000 | RW- |

1. Fetch instruction from segment 0 at address 500
2. Write data to segment 0 at address 100
3. Read from segment 1 at address 2500

<details>
<summary>Show Answer</summary>

```
1. Segment 0 (code), address 500, execute(X)
   - 500 < 2000: Range OK
   - X permission exists: Protection OK
   → Normal execution

2. Segment 0 (code), address 100, write(W)
   - 100 < 2000: Range OK
   - No W permission (R-X only): Protection violation!
   → Protection Fault

3. Segment 1 (data), address 2500, read(R)
   - 2500 < 3000: Range OK
   - R permission exists (RW-): Protection OK
   → Normal execution
```

</details>

### Problem 3: Segment Sharing
Processes A and B both use the same shared library (1000 bytes). When this library is loaded at physical address 0x10000, create segment tables for each process.

(A maps library to segment 2, B to segment 3)

<details>
<summary>Show Answer</summary>

```
Process A Segment Table:
| Segment | Base    | Limit | Protection | Description |
|---------|---------|-------|------------|-------------|
| 0       | 0x5000  | 2000  | R-X        | A's code    |
| 1       | 0x8000  | 1500  | RW-        | A's data    |
| 2       | 0x10000 | 1000  | R-X        | Shared library |

Process B Segment Table:
| Segment | Base    | Limit | Protection | Description |
|---------|---------|-------|------------|-------------|
| 0       | 0x20000 | 3000  | R-X        | B's code    |
| 1       | 0x25000 | 2000  | RW-        | B's data    |
| 2       | ...     | ...   | ...        | Other       |
| 3       | 0x10000 | 1000  | R-X        | Shared library |

→ Segments A:2 and B:3 point to same physical address (0x10000)
→ Library code loaded only once in memory
```

</details>

### Problem 4: External Fragmentation
Explain why external fragmentation occurs with segmentation and how to solve it.

<details>
<summary>Show Answer</summary>

```
Reasons for external fragmentation:
1. Segments are variable size
2. Repeated segment allocation/deallocation creates small holes in memory
3. Total free space sufficient but not contiguous

Example:
[Used][500KB hole][Used][300KB hole][Used][200KB hole]
Total free: 1000KB, but cannot allocate 600KB segment

Solutions:
1. Compaction
   - Move segments to consolidate holes
   - High cost (requires process suspension)

2. Combine with paging
   - Divide segments into pages
   - Eliminate external fragmentation
   - Standard approach in modern systems

3. Buddy system
   - Divide memory into power-of-2 sizes
   - Reduce fragmentation, fast merging
```

</details>

### Problem 5: x86 Segment Selector
A user process's CS register value in Linux is 0x23. Interpret this.

<details>
<summary>Show Answer</summary>

```
0x23 = 0b00100011

Segment selector structure: [Index(13)|TI(1)|RPL(2)]

0x23 = 0000 0000 0010 0011

Index (bits 15-3): 0000 0000 0010 0 = 4
TI (bit 2): 0 = GDT
RPL (bits 1-0): 11 = 3 (user mode)

Interpretation:
- 4th entry in GDT (GDT_ENTRY_DEFAULT_USER_CS)
- User mode (Ring 3)
- This points to user code segment

Matches Linux kernel code:
#define __USER_CS (GDT_ENTRY_DEFAULT_USER_CS * 8 + 3)
              = (4 * 8 + 3)
              = 32 + 3
              = 35
              = 0x23 ✓
```

</details>

---

## Next Steps

Learn about demand paging and virtual memory systems in [14_Virtual_Memory.md](./14_Virtual_Memory.md)!

---

## References

- Silberschatz, "Operating System Concepts" Chapter 8
- Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3
- Linux kernel source: `arch/x86/kernel/cpu/common.c` (GDT initialization)
- Tanenbaum, "Modern Operating Systems" Chapter 3
