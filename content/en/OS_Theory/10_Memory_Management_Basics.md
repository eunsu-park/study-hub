# Memory Management Basics ⭐⭐

**Previous**: [Deadlock](./09_Deadlock.md) | **Next**: [Contiguous Memory Allocation](./11_Contiguous_Memory_Allocation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish logical addresses from physical addresses
2. Explain address binding at compile time, load time, and run time
3. Describe the role of the MMU in address translation
4. Explain swapping and its performance implications
5. Compare different memory allocation strategies

---

Every variable, every function, every data structure in your program lives somewhere in memory. The OS decides where -- and that decision affects performance, security, and whether your program can even run. Memory management is the bridge between your program's abstract view of memory and the physical RAM chips on the motherboard.

## Table of Contents

1. [Need for Memory Management](#1-need-for-memory-management)
2. [Address Binding](#2-address-binding)
3. [Logical and Physical Addresses](#3-logical-and-physical-addresses)
4. [MMU (Memory Management Unit)](#4-mmu-memory-management-unit)
5. [Dynamic Loading](#5-dynamic-loading)
6. [Dynamic Linking](#6-dynamic-linking)
7. [Swapping](#7-swapping)
8. [Practice Problems](#practice-problems)

---

## 1. Need for Memory Management

### Multiprogramming Environment

```
┌─────────────────────────────────────────────────────────────┐
│                        Physical Memory                       │
├─────────────────────────────────────────────────────────────┤
│  Operating System (Kernel)                                   │
├─────────────────────────────────────────────────────────────┤
│  Process A                                                   │
├─────────────────────────────────────────────────────────────┤
│  Process B                                                   │
├─────────────────────────────────────────────────────────────┤
│  Process C                                                   │
├─────────────────────────────────────────────────────────────┤
│  Free Space                                                  │
└─────────────────────────────────────────────────────────────┘
```

### Goals of Memory Management

| Goal | Description |
|------|-------------|
| **Protection** | Protect memory regions between processes |
| **Relocation** | Allow processes to be placed anywhere in memory |
| **Sharing** | Allow multiple processes to share common code |
| **Efficiency** | Minimize memory waste |
| **Logical Organization** | Organize programs in modular units |

---

## 2. Address Binding

Address binding is the process of connecting program instructions and data to memory addresses.

### Binding Time

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Source Code │───▶│ Object Code  │───▶│ Executable   │───▶│   Memory     │
│  (No Address)│    │(Relocatable) │    │ (Loadable)   │    │(Phys Address)│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                         ↑                    ↑                    ↑
                    Compile-time          Load-time           Execution-time
                      Binding              Binding              Binding
```

### 2.1 Compile-time Binding

When the location where a process will be loaded is known at compile time:

```c
// Example using absolute addresses (old MS-DOS)
// Assume program always starts at address 0x1000
#define BASE_ADDRESS 0x1000

int main() {
    int* ptr = (int*)(BASE_ADDRESS + 0x100);  // Absolute address
    *ptr = 42;
    return 0;
}
```

**Characteristics:**
- Generates absolute code
- Requires recompilation if location changes
- Mainly used in embedded systems

### 2.2 Load-time Binding

When process location is unknown until execution:

```
┌─────────────────────────────────────────────────────────────┐
│                     Relocatable Code                         │
├─────────────────────────────────────────────────────────────┤
│  LOAD  R1, [0x100]     ; Relative address 0x100             │
│  ADD   R1, R2                                                │
│  STORE R1, [0x200]     ; Relative address 0x200             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ Loader sets base address 0x5000
┌─────────────────────────────────────────────────────────────┐
│                     After Memory Loading                     │
├─────────────────────────────────────────────────────────────┤
│  LOAD  R1, [0x5100]    ; 0x5000 + 0x100                     │
│  ADD   R1, R2                                                │
│  STORE R1, [0x5200]    ; 0x5000 + 0x200                     │
└─────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- Generates relocatable code
- Loader modifies all addresses
- Cannot move after loading

### 2.3 Execution-time Binding

When process can change memory location during execution:

```
┌──────────────────┐                  ┌──────────────────┐
│   CPU (Logical)  │                  │  Physical Memory │
│                  │                  │                  │
│   Address: 0x100 │─────┐            │                  │
└──────────────────┘     │            │                  │
                         ▼            │                  │
                 ┌──────────────┐     │  ┌────────────┐ │
                 │     MMU      │     │  │ 0x5100     │◀┤
                 │              │     │  │ (Actual)   │ │
                 │ Base: 0x5000│────▶│  └────────────┘ │
                 │              │     │                  │
                 │ 0x100+0x5000│     │                  │
                 │  = 0x5100   │     │                  │
                 └──────────────┘     └──────────────────┘
```

**Characteristics:**
- Requires hardware support (MMU)
- Standard method in modern OSes
- Allows process movement (swapping)

---

## 3. Logical and Physical Addresses

### 3.1 Concept Comparison

| Category | Logical Address | Physical Address |
|----------|----------------|------------------|
| **Alias** | Virtual Address | Real Address |
| **Generated by** | CPU | Memory device recognizes |
| **Range** | 0 ~ Process size | 0 ~ Physical memory size |
| **Programmer** | Uses | No need to know |

### 3.2 Address Spaces

```
    Process A's              Process B's                 Physical Memory
    Logical Address Space    Logical Address Space

┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ 0x0000       │          │ 0x0000       │          │ 0x0000 OS    │
│              │          │              │          ├──────────────┤
│ Code         │──────────┼──────────────┼─────────▶│ 0x1000 A Code│
│              │          │ Code         │──┐       ├──────────────┤
├──────────────┤          │              │  │       │ 0x2000 A Data│
│ Data         │──────────┼──────────────┼──┼──────▶│              │
│              │          ├──────────────┤  │       ├──────────────┤
├──────────────┤          │ Data         │──┼──────▶│ 0x3000 B Code│
│ Heap         │          │              │  │       ├──────────────┤
│              │          ├──────────────┤  │       │ 0x4000 B Data│
├──────────────┤          │ Heap         │  │       ├──────────────┤
│              │          │              │  │       │ 0x5000 A Heap│
│ (Free)       │          ├──────────────┤  │       ├──────────────┤
│              │          │              │  │       │ 0x6000 B Heap│
├──────────────┤          │              │  │       ├──────────────┤
│ Stack        │          ├──────────────┤  │       │              │
│ 0xFFFF       │          │ Stack        │  │       │ Free         │
└──────────────┘          │ 0xFFFF       │  │       │              │
                          └──────────────┘  │       └──────────────┘
                                            │
                                   MMU performs address translation
```

---

## 4. MMU (Memory Management Unit)

### 4.1 Basic Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                            CPU                                   │
│  ┌─────────────┐                                                │
│  │   Program   │                                                │
│  │   Counter   │──▶ Logical Address: 0x1234                     │
│  └─────────────┘                                                │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                           MMU                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                                                              │ │
│  │   Logical Address    Relocation Register    Physical Address│ │
│  │     0x1234       +       0x8000        =       0x9234       │ │
│  │                                                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │   Limit Register: 0x4000                                     │ │
│  │   Logical 0x1234 < 0x4000 ? ──▶ OK (Access allowed)         │ │
│  │   Logical 0x5000 < 0x4000 ? ──▶ TRAP! (Protection violation)│ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Physical Memory                            │
│                                                                   │
│                        Access address 0x9234                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Dynamic Loading

### 5.1 Concept

A technique that loads program code into memory only when needed, rather than loading all code at once.

---

## 6. Dynamic Linking

### 6.1 Static Linking vs Dynamic Linking

Static linking duplicates library code in each program. Dynamic linking shares a single library copy across all programs.

---

## 7. Swapping

### 7.1 Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                        Swapping Process                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Memory (RAM)                      Disk (Backing Store)        │
│  ┌──────────────┐                  ┌──────────────┐             │
│  │ OS           │                  │              │             │
│  ├──────────────┤                  │              │             │
│  │ Process A    │  ──Swap Out──▶  │ Process A    │             │
│  ├──────────────┤                  │ (Image)      │             │
│  │ Process B    │                  │              │             │
│  ├──────────────┤  ◀──Swap In───  │ Process C    │             │
│  │ Process C    │                  │ (Image)      │             │
│  │ (New loaded) │                  │              │             │
│  └──────────────┘                  └──────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

### Problem 1: Address Translation
Given relocation register 0x4000 and limit register 0x3000:
1. What is the physical address of logical address 0x1500?
2. What happens when accessing logical address 0x3500?

<details>
<summary>Show Answer</summary>

1. Physical address = 0x4000 + 0x1500 = 0x5500
2. 0x3500 >= 0x3000 (exceeds limit) → Segmentation Fault

</details>

---

## Next Steps

Learn about memory partitioning and allocation strategies in [11_Contiguous_Memory_Allocation.md](./11_Contiguous_Memory_Allocation.md)!

---

## References

- Silberschatz, "Operating System Concepts" Chapter 8
- Tanenbaum, "Modern Operating Systems" Chapter 3
- Linux man pages: `dlopen(3)`, `mmap(2)`
- `/proc/[pid]/maps` - Check process memory map
