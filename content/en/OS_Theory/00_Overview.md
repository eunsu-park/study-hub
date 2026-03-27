# Operating System Theory Learning Guide

## Introduction

This folder contains materials for systematically learning Operating System (OS) theory. You can progressively learn core OS concepts from process management to memory management and file systems.

**Target Audience**: Developers with C/C++ programming experience, those studying CS fundamentals

---

## Learning Roadmap

```
[OS Basics]              [CPU/Synchronization]       [Memory/Files]
     │                          │                           │
     ▼                          ▼                           ▼
OS Overview ─────────▶ CPU Scheduling Basics ────▶ Memory Management Basics
     │                          │                           │
     ▼                          ▼                           ▼
Process Concepts ────────▶ Scheduling Algorithms ────▶ Virtual Memory
     │                          │                           │
     ▼                          ▼                           ▼
Threads/Multithreading ──▶ Advanced Scheduling ────────▶ Page Replacement
     │                          │                           │
     ▼                          ▼                           ▼
     └──────────────▶ Synchronization Basics ─────▶ File Systems
                            │                           │
                            ▼                           ▼
                       Synchronization Tools ──────▶ I/O Systems
                            │
                            ▼
                        Deadlock
```

---

## Prerequisites

- **C/C++ Programming**: Pointers, memory management, multithreading basics
- **Computer Architecture Basics**: CPU, memory hierarchy, interrupts
- **Basic Data Structures**: Queue, stack, linked list
- **Basic Algorithms**: Complexity analysis (Big O)

---

## File List

### OS Basics (01-03)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [01_OS_Overview.md](./01_OS_Overview.md) | ⭐ | OS definition, roles, history, kernel architecture |
| [02_Process_Concepts.md](./02_Process_Concepts.md) | ⭐⭐ | Process memory structure, PCB, state transitions |
| [03_Threads_and_Multithreading.md](./03_Threads_and_Multithreading.md) | ⭐⭐ | Thread vs process, multithreading models |

### CPU Scheduling (04-06)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [04_CPU_Scheduling_Basics.md](./04_CPU_Scheduling_Basics.md) | ⭐⭐ | CPU/I/O burst, scheduling goals, scheduler types |
| [05_Scheduling_Algorithms.md](./05_Scheduling_Algorithms.md) | ⭐⭐⭐ | FCFS, SJF, SRTF, Priority, RR, Gantt charts |
| [06_Advanced_Scheduling.md](./06_Advanced_Scheduling.md) | ⭐⭐⭐ | MLFQ, multiprocessor scheduling, real-time scheduling |

### Process Synchronization (07-09)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [07_Synchronization_Basics.md](./07_Synchronization_Basics.md) | ⭐⭐⭐ | Race conditions, critical sections, Peterson's Solution |
| [08_Synchronization_Tools.md](./08_Synchronization_Tools.md) | ⭐⭐⭐ | Mutex, semaphore, monitor, classic synchronization problems |
| [09_Deadlock.md](./09_Deadlock.md) | ⭐⭐⭐ | Deadlock conditions, prevention, avoidance, detection, Banker's algorithm |

### Memory Management (10-13)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [10_Memory_Management_Basics.md](./10_Memory_Management_Basics.md) | ⭐⭐ | Address binding, swapping, memory allocation overview |
| [11_Contiguous_Memory_Allocation.md](./11_Contiguous_Memory_Allocation.md) | ⭐⭐⭐ | First-fit, best-fit, fragmentation, compaction |
| [12_Paging.md](./12_Paging.md) | ⭐⭐⭐ | Page table, TLB, multilevel paging |
| [13_Segmentation.md](./13_Segmentation.md) | ⭐⭐⭐ | Segment table, comparison with paging |

### Virtual Memory (14-15)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [14_Virtual_Memory.md](./14_Virtual_Memory.md) | ⭐⭐⭐ | Demand paging, page fault, valid/invalid bit |
| [15_Page_Replacement.md](./15_Page_Replacement.md) | ⭐⭐⭐ | FIFO, LRU, LFU, Clock, thrashing |

### File Systems and I/O (16-18)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [16_File_System_Basics.md](./16_File_System_Basics.md) | ⭐⭐ | File concepts, directory structure, access methods |
| [17_File_System_Implementation.md](./17_File_System_Implementation.md) | ⭐⭐⭐ | Allocation methods, FAT, inode, journaling |
| [18_IO_and_IPC.md](./18_IO_and_IPC.md) | ⭐⭐⭐ | I/O hardware, DMA, IPC communication |

### Advanced Topics (20-28)

| File | Difficulty | Key Topics |
|------|-----------|-----------|
| [20_Advanced_Virtual_Memory.md](./20_Advanced_Virtual_Memory.md) | ⭐⭐⭐ | Advanced paging, memory-mapped files, copy-on-write |
| [21_Disk_Scheduling_and_IO.md](./21_Disk_Scheduling_and_IO.md) | ⭐⭐⭐ | Disk scheduling algorithms, I/O optimization |
| [22_Real_Time_OS.md](./22_Real_Time_OS.md) | ⭐⭐⭐ | Real-time scheduling, hard/soft real-time, RTOS design |
| [23_Container_Internals.md](./23_Container_Internals.md) | ⭐⭐⭐ | Namespaces, cgroups, container runtimes |
| [24_eBPF_and_Kernel_Tracing.md](./24_eBPF_and_Kernel_Tracing.md) | ⭐⭐⭐⭐ | eBPF programs, tracepoints, kernel observability |
| [25_OS_Security.md](./25_OS_Security.md) | ⭐⭐⭐ | Access control, capabilities, security modules |
| [26_Microkernel_Design.md](./26_Microkernel_Design.md) | ⭐⭐⭐⭐ | Microkernel architecture, IPC, L4, seL4 |
| [27_Modern_Schedulers.md](./27_Modern_Schedulers.md) | ⭐⭐⭐⭐ | CFS, BFS, EEVDF, scheduler design trade-offs |
| [28_Capstone_xv6_Walkthrough.md](./28_Capstone_xv6_Walkthrough.md) | ⭐⭐⭐⭐ | xv6 source walkthrough, boot, syscalls, file system |

---

## Recommended Learning Order

### Stage 1: OS Basics (1 week)
```
01_OS_Overview → 02_Process_Concepts → 03_Threads_and_Multithreading
```
Understand basic OS concepts and differences between processes/threads.

### Stage 2: CPU Scheduling (1-2 weeks)
```
04_CPU_Scheduling_Basics → 05_Scheduling_Algorithms → 06_Advanced_Scheduling
```
Learn CPU scheduling goals and various algorithms.

### Stage 3: Process Synchronization (1-2 weeks)
```
07_Synchronization_Basics → 08_Synchronization_Tools → 09_Deadlock
```
Study concurrency issues and solutions in depth.

### Stage 4: Memory Management (1-2 weeks)
```
10_Memory_Management_Basics → 11_Contiguous_Memory_Allocation → 12_Paging → 13_Segmentation
```

### Stage 5: Virtual Memory (1 week)
```
14_Virtual_Memory → 15_Page_Replacement
```

### Stage 6: Files/I/O (1 week)
```
16_File_System_Basics → 17_File_System_Implementation → 18_IO_and_IPC
```

### Stage 7: Advanced Topics (2-3 weeks)
```
20_Advanced_Virtual_Memory → 21_Disk_Scheduling_and_IO → 22_Real_Time_OS
→ 23_Container_Internals → 24_eBPF_and_Kernel_Tracing → 25_OS_Security
→ 26_Microkernel_Design → 27_Modern_Schedulers → 28_Capstone_xv6_Walkthrough
```
Deep dive into modern OS topics: containers, eBPF tracing, security, microkernel design, and scheduler internals. The capstone walks through xv6 source code to connect all concepts.

---

## Practice Environment

### Required Tools

```bash
# Linux environment (recommended)
# Ubuntu, Fedora, or macOS

# GCC compiler
gcc --version
g++ --version

# pthread library (multithreading)
# Included by default on Linux

# Process monitoring
ps aux
top
htop
```

### System Information

```bash
# CPU information
cat /proc/cpuinfo

# Memory information
cat /proc/meminfo
free -h

# Process status
cat /proc/[PID]/status
```

---

## Core Concepts Quick Reference

### Process State Transitions

```
         New
          │
          ▼
       ┌─────┐  Dispatch   ┌─────┐
       │Ready│───────────▶│Run  │
       │     │◀───────────│     │
       └─────┘  Timeout    └─────┘
          ▲                   │
          │     I/O Complete │ I/O Request
          │                   ▼
          │              ┌─────┐
          └──────────────│Wait │
                         │     │
                         └─────┘
```

### Scheduling Algorithm Comparison

| Algorithm | Preemptive | Starvation | Features |
|-----------|-----------|-----------|----------|
| FCFS | Non-preemptive | None | Simple, convoy effect |
| SJF | Non-preemptive | Possible | Optimal average waiting time |
| SRTF | Preemptive | Possible | Preemptive SJF |
| Priority | Both | Possible | Solve with aging |
| RR | Preemptive | None | Time-sharing, time quantum important |
| MLFQ | Preemptive | Possible | Adaptive, practical |

### Synchronization Tool Comparison

| Tool | Value Range | Use Case |
|------|------------|----------|
| Mutex | 0/1 | Mutual exclusion |
| Binary Semaphore | 0/1 | Mutual exclusion |
| Counting Semaphore | 0~N | Resource counting |
| Monitor | - | Advanced synchronization |

---

## Related Materials

### Links with Other Folders

| Folder | Related Content |
|--------|----------------|
| [Linux/](../Linux/00_Overview.md) | Linux system programming, process management |
| [Computer_Architecture/](../Computer_Architecture/00_Overview.md) | CPU architecture, memory hierarchy, interrupts |
| [C_Basics/](../C_Basics/00_Overview.md) | System calls, multithreading programming |
| [Networking/](../Networking/00_Overview.md) | Socket programming, I/O models |

### External Resources

- [Operating System Concepts (Dinosaur Book)](https://www.os-book.com/)
- [OSTEP (Free online textbook)](https://pages.cs.wisc.edu/~remzi/OSTEP/)
- [MIT 6.828: Operating System Engineering](https://pdos.csail.mit.edu/6.828/)
- [Linux Kernel Development (Robert Love)](https://www.oreilly.com/library/view/linux-kernel-development/9780768696974/)

---

## Learning Tips

1. **Practice is Essential**: Write and execute code yourself
2. **Visualization**: Draw process states, scheduling Gantt charts yourself
3. **Use Linux**: Check real system status with /proc filesystem
4. **Step-by-Step Learning**: Fully understand basic concepts before moving to next stage
5. **Problem Solving**: Make sure to solve practice problems in each lesson
