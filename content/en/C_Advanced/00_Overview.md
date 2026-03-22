# C Advanced

## Introduction

This topic covers advanced C programming: pointer mastery, systems programming, data structures, concurrency, and cross-platform development.

**Prerequisites**: [C_Basics](../C_Basics/00_Overview.md) (or equivalent knowledge of C fundamentals including pointers, structs, and dynamic memory)

---

## Learning Roadmap

```
[Pointers & Data Structures]        [Systems Programming]         [Tooling & Platform]
  |                                    |                              |
  v                                    v                              v
Advanced Pointers ----+          Process Mgmt -----+           Embedded Systems
  |                   |            |                |              |
  v                   |            v                |              v
Memory Management     |          Mini Shell         |           Debugging & Profiling
  |                   |            |                |              |
  v                   |            v                |              v
Dynamic Array         |          Multithreading     |           Cross-Platform Dev
  |                   |            |                |              |
  v                   |            v                |              v
Linked List           |          Network Prog       |           Snake Game (capstone)
  |                   |            |                |
  v                   |            v                |
Stack & Queue         |          IPC & Signals -----+
  |                   |
  v                   |
Hash Table            |
  |                   |
  v                   |
File Encryption ------+
```

---

## File List

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| [01](./01_Advanced_Pointers.md) | Advanced Pointers | ⭐⭐⭐ | function pointers, void*, pointer arrays, const correctness |
| [02](./02_Advanced_Memory_Management.md) | Advanced Memory Management | ⭐⭐⭐ | memory layout, mmap, custom allocators, memory pools |
| [03](./03_Bit_Operations.md) | Bit Operations | ⭐⭐ | bitwise operators, bit masking, register manipulation |
| [04](./04_Project_Dynamic_Array.md) | Project: Dynamic Array | ⭐⭐ | malloc/realloc, growable arrays, amortized cost |
| [05](./05_Project_Linked_List.md) | Project: Linked List | ⭐⭐⭐ | singly/doubly linked, insertion, deletion, reversal |
| [06](./06_Project_Stack_Queue.md) | Project: Stack and Queue | ⭐⭐ | LIFO/FIFO, array and linked implementations |
| [07](./07_Project_Hash_Table.md) | Project: Hash Table | ⭐⭐⭐ | hash functions, chaining, open addressing |
| [08](./08_Project_File_Encryption.md) | Project: File Encryption | ⭐⭐ | XOR cipher, byte-level file processing |
| [09](./09_Process_Management.md) | Process Management | ⭐⭐⭐ | fork, exec, wait, process lifecycle |
| [10](./10_Project_Mini_Shell.md) | Project: Mini Shell | ⭐⭐⭐⭐ | shell implementation, pipes, redirection |
| [11](./11_Multithreading.md) | Multithreading | ⭐⭐⭐⭐ | pthreads, mutexes, condition variables, thread pool |
| [12](./12_Network_Programming.md) | Network Programming | ⭐⭐⭐⭐ | TCP/UDP sockets, client-server, select/poll |
| [13](./13_IPC_and_Signals.md) | IPC and Signals | ⭐⭐⭐⭐ | pipes, shared memory, message queues, signals |
| [14](./14_Embedded_Systems.md) | Embedded Systems | ⭐⭐⭐ | GPIO, serial, I2C/SPI, volatile, register access |
| [15](./15_Debugging_and_Profiling.md) | Debugging and Profiling | ⭐⭐⭐ | GDB advanced, Valgrind, ASan, gprof, unit testing |
| [16](./16_Cross_Platform_Development.md) | Cross-Platform Development | ⭐⭐⭐ | portability, CMake, platform abstraction |
| [17](./17_Project_Snake_Game.md) | Project: Snake Game | ⭐⭐⭐ | terminal control, game loop, ncurses |

---

## Recommended Learning Order

### Path 1: Pointers & Data Structures
1. Advanced Pointers -> Memory Management -> Dynamic Array -> Linked List -> Stack & Queue -> Hash Table -> File Encryption

### Path 2: Systems Programming
2. Process Management -> Mini Shell -> Multithreading -> Network Programming -> IPC & Signals

### Path 3: Tooling & Platform
3. Embedded Systems -> Debugging & Profiling -> Cross-Platform Development -> Snake Game (capstone)

---

## Practice Environment

```bash
# Check GCC version (C11 support required)
gcc --version

# Compile with warnings and debug info
gcc -Wall -Wextra -std=c11 -g program.c -o program

# Run with Valgrind (Linux/macOS)
valgrind --leak-check=full ./program

# Compile with AddressSanitizer
gcc -fsanitize=address -g program.c -o program
```

---

## Related Materials

- [C_Basics/](../C_Basics/00_Overview.md) - C fundamentals (variables, control flow, functions, basic pointers)
- [Linux/](../Linux/00_Overview.md) - Linux environment and shell scripting
- [OS_Theory/](../OS_Theory/00_Overview.md) - Operating system concepts (processes, memory, scheduling)
- [Computer_Architecture/](../Computer_Architecture/00_Overview.md) - Hardware fundamentals
- [Algorithm/](../Algorithm/00_Overview.md) - Data structures and algorithms
- [Networking/](../Networking/00_Overview.md) - Network protocols and architecture
