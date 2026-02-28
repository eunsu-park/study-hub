# C Programming Learning Guide

## Introduction

This folder contains materials for systematically learning C programming. From basic syntax to embedded systems, you can learn step-by-step through hands-on projects.

**Target Audience**: Programming beginners ~ intermediate learners

---

## Learning Roadmap

```
[Basics]         [Intermediate]      [Advanced]       [Embedded]
  │                │                   │                  │
  ▼                ▼                   ▼                  ▼
Setup ──────▶ Dynamic Array ───▶ Snake Game ────▶ Embedded Basics
  │           │                   │              │
  ▼           ▼                   ▼              ▼
Review ─────▶ Linked List ───▶ Mini Shell ───▶ Bit Operations
  │           │                   │              │
  ▼           ▼                   ▼              ▼
Calculator ─▶ File Encrypt ──▶ Multithreading ▶ GPIO Control
  │           │                                  │
  ▼           ▼                                  ▼
Guessing ───▶ Stack & Queue                  Serial Comm
  │           │
  ▼           ▼
Address Book ▶ Hash Table
```

---

## Prerequisites

- Basic computer usage skills
- Terminal/command-line experience (recommended)
- Text editor or IDE usage

---

## File List

| Filename | Difficulty | Key Content |
|--------|--------|----------|
| [C Language Environment Setup](./01_Environment_Setup.md) | ⭐ | Development environment setup, compiler installation |
| [C Language Basics Quick Review](./02_C_Basics_Review.md) | ⭐ | Variables, data types, operators, control structures, functions |
| [Project 1: Basic Arithmetic Calculator](./03_Project_Calculator.md) | ⭐ | Functions, switch-case, scanf |
| [Project 2: Number Guessing Game](./04_Project_Number_Guessing.md) | ⭐ | Loops, random numbers, conditionals |
| [Project 3: Address Book Program](./05_Project_Address_Book.md) | ⭐⭐ | Structures, arrays, file I/O |
| [Project 4: Dynamic Array](./06_Project_Dynamic_Array.md) | ⭐⭐ | malloc, realloc, free |
| [Project 5: Linked List](./07_Project_Linked_List.md) | ⭐⭐⭐ | Pointers, dynamic data structures |
| [Project 6: File Encryption Tool](./08_Project_File_Encryption.md) | ⭐⭐ | File processing, bit operations |
| [Project 7: Stack and Queue](./09_Project_Stack_Queue.md) | ⭐⭐ | Data structures, LIFO/FIFO |
| [Project 8: Hash Table](./10_Project_Hash_Table.md) | ⭐⭐⭐ | Hashing, collision handling |
| [Project 10: Terminal Snake Game](./11_Project_Snake_Game.md) | ⭐⭐⭐ | Terminal control, game loop |
| [Project 11: Mini Shell](./12_Project_Mini_Shell.md) | ⭐⭐⭐⭐ | fork, exec, pipes |
| [Project 12: Multithreaded Programming](./13_Project_Multithreading.md) | ⭐⭐⭐⭐ | pthread, synchronization |
| [Embedded Programming Basics](./14_Embedded_Basics.md) | ⭐ | Arduino, GPIO basics |
| [Advanced Bit Operations](./15_Bit_Operations.md) | ⭐⭐ | Bit masking, registers |
| [Project 15: GPIO Control](./16_Project_GPIO_Control.md) | ⭐⭐ | LED, button, debouncing |
| [Project 16: Serial Communication](./17_Project_Serial_Communication.md) | ⭐⭐ | UART, command parsing |
| [Debugging and Memory Analysis](./18_Debugging_Memory_Analysis.md) | ⭐⭐⭐ | GDB, Valgrind, AddressSanitizer |
| [Advanced Embedded Protocols](./19_Advanced_Embedded_Protocols.md) | ⭐⭐⭐ | PWM, I2C, SPI, ADC |
| [Advanced C Pointers](./20_Advanced_Pointers.md) | ⭐⭐⭐ | Pointer arithmetic, multi-level pointers, function pointers, dynamic memory |
| [Network Programming in C](./21_Network_Programming.md) | ⭐⭐⭐⭐ | TCP/UDP sockets, client-server, I/O multiplexing (select/poll) |
| [Inter-Process Communication and Signals](./22_IPC_and_Signals.md) | ⭐⭐⭐⭐ | Pipes, shared memory, message queues, signal handling |
| [Testing and Profiling in C](./23_Testing_and_Profiling.md) | ⭐⭐⭐ | Unit testing (Unity, assert.h), gprof, Valgrind, optimization |
| [Cross-Platform Development](./24_Cross_Platform_Development.md) | ⭐⭐⭐ | Portability, conditional compilation, CMake, platform abstraction |

---

## Recommended Learning Path

### Beginner (C Introduction)
1. Environment Setup → Basics Review → Calculator → Number Guessing → Address Book

### Intermediate (Data Structures & Pointers)
2. Advanced Pointers → Dynamic Array → Linked List → File Encryption → Stack & Queue → Hash Table

### Advanced (Systems Programming)
3. Snake Game → Mini Shell → Multithreading → Network Programming → IPC & Signals

### Embedded (Arduino)
4. Embedded Basics → Bit Operations → GPIO Control → Serial Communication → Advanced Embedded Protocols

### Debugging (Optional)
5. Debugging and Memory Analysis (recommended after completing all courses)

---

## Related Materials

- [Docker Learning](../Docker/00_Overview.md) - Development environment containerization
- [Git Learning](../Git/00_Overview.md) - Version control
