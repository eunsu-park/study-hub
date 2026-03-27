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

### 4.2 Relocation and Limit Registers

```c
// MMU operation pseudocode
typedef struct {
    uint32_t relocation_register;  // Relocation register (base address)
    uint32_t limit_register;       // Limit register (process size)
} MMU;

uint32_t translate_address(MMU* mmu, uint32_t logical_address) {
    // 1. Bounds check
    if (logical_address >= mmu->limit_register) {
        // Protection violation! Raise trap
        raise_trap(SEGMENTATION_FAULT);
        return 0;
    }

    // 2. Address translation
    uint32_t physical_address = logical_address + mmu->relocation_register;

    return physical_address;
}

// Example
MMU mmu = {
    .relocation_register = 0x8000,  // Process starts at 0x8000
    .limit_register = 0x4000        // Process size 16KB
};

// Logical address 0x1234 → Physical address 0x9234 (OK)
// Logical address 0x5000 → SEGMENTATION FAULT (exceeds limit)
```

### 4.3 Context Switching and MMU

```
┌───────────────────────────────────────────────────────────────────┐
│                     Context Switching Process                      │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Process A running             Switching to Process B            │
│   ┌──────────────┐                                                │
│   │ MMU Registers│            1. Save A's state                   │
│   │ Base: 0x8000 │               - CPU registers                  │
│   │ Limit: 0x4000│               - MMU settings                   │
│   └──────────────┘                                                │
│           │                   2. Restore B's state                │
│           │                      - CPU registers                  │
│           ▼                      - MMU settings                   │
│   ┌──────────────┐                                                │
│   │ MMU Registers│            3. Resume execution                 │
│   │ Base: 0x14000│                                                │
│   │ Limit: 0x6000│                                                │
│   └──────────────┘                                                │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

When the OS performs a context switch, it must update the MMU's relocation and limit registers to reflect the new process's memory region. If these registers are not updated, the new process would access the previous process's memory -- a critical security violation.

---

## 5. Dynamic Loading

### 5.1 Concept

A technique that loads program code into memory only when needed, rather than loading all code at once. This is sometimes called **lazy loading** because routines remain on disk until they are actually called.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dynamic Loading Process                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Program starts: only main() is loaded                       │
│                                                                  │
│     Memory                    Disk                              │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │ main()       │         │ func_A()     │                     │
│  │              │         │ func_B()     │                     │
│  │              │         │ func_C()     │                     │
│  └──────────────┘         └──────────────┘                     │
│                                                                  │
│  2. When func_A() is called: loaded from disk                   │
│                                                                  │
│     Memory                    Disk                              │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │ main()       │    ◀──  │ func_A()     │                     │
│  │ func_A()     │         │ func_B()     │                     │
│  │              │         │ func_C()     │                     │
│  └──────────────┘         └──────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Implementation with dlopen/dlsym

On Unix-like systems, dynamic loading is implemented using the `dlopen()` and `dlsym()` API:

```c
#include <stdio.h>
#include <dlfcn.h>  // Header for dynamic loading

// Define function pointer type
typedef int (*MathFunc)(int, int);

int main() {
    void* handle;
    MathFunc add_func;
    char* error;

    // 1. Dynamically load library
    handle = dlopen("./libmath.so", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Load failed: %s\n", dlerror());
        return 1;
    }

    // 2. Look up function symbol
    dlerror();  // Clear previous errors
    add_func = (MathFunc)dlsym(handle, "add");
    error = dlerror();
    if (error != NULL) {
        fprintf(stderr, "Symbol lookup failed: %s\n", error);
        return 1;
    }

    // 3. Call the function
    printf("Result: %d\n", add_func(10, 20));  // Output: Result: 30

    // 4. Unload library
    dlclose(handle);

    return 0;
}
```

```bash
# Build shared library
gcc -shared -fPIC -o libmath.so math.c

# Compile main program (link dynamic loading library)
gcc -o main main.c -ldl

# Run
./main
```

### 5.3 Advantages and Disadvantages

| Advantage | Disadvantage |
|-----------|-------------|
| Reduced memory usage | Latency on first call |
| Unused code is never loaded | Increased implementation complexity |
| No special OS support required | Error handling needed |

---

## 6. Dynamic Linking

### 6.1 Static Linking vs Dynamic Linking

```
┌─────────────────────────────────────────────────────────────────┐
│                        Static Linking                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Program A          Program B          Program C              │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐          │
│  │ Code     │        │ Code     │        │ Code     │          │
│  ├──────────┤        ├──────────┤        ├──────────┤          │
│  │ libc     │        │ libc     │        │ libc     │          │
│  │ (copy)   │        │ (copy)   │        │ (copy)   │          │
│  └──────────┘        └──────────┘        └──────────┘          │
│                                                                  │
│  Problem: Library code duplicated 3 times!                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Dynamic Linking                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Program A    Program B    Program C        Shared Library     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     ┌──────────┐     │
│  │ Code     │  │ Code     │  │ Code     │     │ libc.so  │     │
│  ├──────────┤  ├──────────┤  ├──────────┤  ┌──│          │     │
│  │ stub     │──┼──────────┼──┼──────────┼──┤  │ printf() │     │
│  └──────────┘  │ stub     │──┼──────────┼──┤  │ malloc() │     │
│                └──────────┘  │ stub     │──┘  │ ...      │     │
│                              └──────────┘     └──────────┘     │
│                                                                  │
│  Advantage: Only one copy of library code in memory!            │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Stub Mechanism

When a program calls a dynamically linked function for the first time, the call goes through a **stub** -- a small piece of code that locates and loads the real function:

```c
// Dynamic linking stub operation (pseudocode)

// On first call
void printf_stub() {
    // 1. Check if library is in memory
    if (!is_library_loaded("libc.so")) {
        // 2. If not, load it
        load_library("libc.so");
    }

    // 3. Obtain actual function address
    void* real_printf = get_symbol_address("printf");

    // 4. Replace stub with actual address (next call jumps directly)
    replace_stub_with_address(real_printf);

    // 5. Call the actual function
    jump_to(real_printf);
}
```

On subsequent calls, the stub has been replaced with a direct jump to the real function, so there is no additional overhead.

### 6.3 Shared Library Version Management

Shared libraries use versioning to maintain backward compatibility. On Linux, the `soname` convention encodes version information:

```bash
# Check shared libraries used by an executable
$ ldd /bin/ls
    linux-vdso.so.1 (0x00007ffd12345000)
    libselinux.so.1 => /lib/x86_64-linux-gnu/libselinux.so.1
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
    /lib64/ld-linux-x86-64.so.2 (0x00007f1234567000)

# Check shared libraries loaded in memory
$ cat /proc/self/maps | grep "\.so"
```

The dynamic linker (`ld-linux.so`) resolves library paths at load time, enabling version upgrades without recompiling applications.

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

### 7.2 Swap Time Calculation

Swap time is dominated by disk transfer time. Understanding swap cost is critical for evaluating system responsiveness:

```
Swap time = Transfer time + Seek time + Rotational latency

Example:
- Process size: 100MB
- Disk transfer rate: 50MB/sec
- Average seek time: 8ms
- Average rotational latency: 4ms

Transfer time = 100MB / 50MB/sec = 2 seconds
Swap out time = 2s + 8ms + 4ms ≈ 2.01 seconds
Swap in time  = 2s + 8ms + 4ms ≈ 2.01 seconds
Total swap time ≈ 4.02 seconds

→ Very slow! Must be minimized.
```

Because swap time is proportional to process size, modern systems prefer swapping individual pages (demand paging) rather than entire processes.

### 7.3 Linux Swap Management

```bash
# Check swap status
$ free -h
              total        used        free      shared  buff/cache   available
Mem:           15Gi       8.0Gi       2.5Gi       500Mi       5.0Gi       6.5Gi
Swap:          8.0Gi       1.2Gi       6.8Gi

# Check swap partitions/files
$ swapon --show
NAME      TYPE      SIZE   USED PRIO
/dev/sda2 partition 8G     1.2G   -2

# Create a swap file (e.g., 4GB)
$ sudo fallocate -l 4G /swapfile
$ sudo chmod 600 /swapfile
$ sudo mkswap /swapfile
$ sudo swapon /swapfile

# Swappiness setting (0-100; higher = more aggressive swapping)
$ cat /proc/sys/vm/swappiness
60

# Change swappiness
$ sudo sysctl vm.swappiness=10
```

### 7.4 Mobile Systems and Swapping

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mobile OS Memory Management                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Characteristics:                                                │
│  - Traditional swapping not used (flash memory lifespan)        │
│  - Instead, apps are terminated and restarted                   │
│                                                                  │
│  iOS:                                                           │
│  - Terminates background apps when memory is low               │
│  - Apps save state before termination, restore on relaunch     │
│                                                                  │
│  Android:                                                       │
│  - zRAM (compressed swap in RAM)                                │
│  - Low Memory Killer (LMK)                                      │
│  - Terminates processes by priority                             │
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

### Problem 2: Dynamic Loading vs Dynamic Linking
Determine whether each description refers to dynamic loading or dynamic linking.

1. Multiple programs share the same library code
2. Can be implemented in user programs without special OS support
3. Uses stub code
4. Loads routines into memory when they are needed

<details>
<summary>Show Answer</summary>

1. Dynamic linking -- shared library usage
2. Dynamic loading -- programmer can implement directly
3. Dynamic linking -- stubs locate the real function address
4. Dynamic loading -- loaded at call time

</details>

### Problem 3: Swap Time Calculation
Calculate the swap-out time for a process given the following conditions:
- Process size: 200MB
- Disk transfer rate: 100MB/sec
- Seek time: 10ms
- Rotational latency: 5ms

<details>
<summary>Show Answer</summary>

Transfer time = 200MB / 100MB/sec = 2 seconds = 2000ms
Swap-out time = 2000ms + 10ms + 5ms = 2015ms ≈ 2.015 seconds

</details>

### Problem 4: Code Analysis
Find the problems in the following code and fix them.

```c
void* handle = dlopen("./plugin.so", RTLD_LAZY);
void (*func)() = dlsym(handle, "process");
func();
dlclose(handle);
```

<details>
<summary>Show Answer</summary>

Problems:
1. No NULL check after `dlopen()` failure
2. No NULL check after `dlsym()` failure

Fixed code:
```c
void* handle = dlopen("./plugin.so", RTLD_LAZY);
if (!handle) {
    fprintf(stderr, "Error: %s\n", dlerror());
    return;
}

dlerror();  // Clear previous errors
void (*func)() = dlsym(handle, "process");
char* error = dlerror();
if (error != NULL) {
    fprintf(stderr, "Error: %s\n", error);
    dlclose(handle);
    return;
}

func();
dlclose(handle);
```

</details>

### Problem 5: System Design
Explain why embedded systems use compile-time binding.

<details>
<summary>Show Answer</summary>

1. **Deterministic behavior**: No execution-time variation, suitable for real-time systems
2. **Reduced overhead**: Direct physical address usage without MMU improves performance
3. **Memory constraints**: MMU hardware is absent or limited
4. **Single program**: Multitasking is unnecessary
5. **Cost reduction**: Simple hardware is sufficient

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
