# C Basics

This topic covers fundamental C programming from environment setup through pointers, structs, and file I/O, culminating in hands-on mini-projects. Whether you are learning your first compiled language or transitioning from a higher-level language like Python, these lessons will give you a solid understanding of how C works close to the hardware and why it remains indispensable in systems programming, embedded development, and performance-critical applications.

## What You'll Learn

This topic provides hands-on coverage of:

- **Getting Started**: Compiler installation, toolchain setup, and your first compiled program
- **Core Language**: Variables, data types, operators, expressions, and control flow
- **Functions**: Declaration, definition, scope rules, recursion, and storage classes
- **Arrays and Strings**: Fixed-size arrays, multidimensional arrays, null-terminated strings, and standard library string functions
- **Pointers**: Address-of and dereference operators, pointer arithmetic, arrays and pointers, pass-by-reference
- **Structs and Unions**: Composite data types, typedef, enumerations, and bit fields
- **Memory Management**: Dynamic allocation with malloc/calloc/realloc/free and avoiding memory leaks
- **File I/O**: Reading and writing text and binary files with the standard I/O library
- **Build Tools**: Preprocessor directives, header guards, Makefiles, compiler flags, and debugging with GDB
- **Projects**: Three guided projects that integrate concepts from earlier lessons

## Prerequisites

- [Programming](../Programming/00_Overview.md) — Familiarity with general programming concepts (variables, control flow, functions)

No prior C experience is required. If you understand what a variable, loop, and function are in any language, you are ready.

## Learning Roadmap

```
                            C Basics — Learning Path
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                                                                             │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
  │  │ 01 Environment│──▶│ 02 Variables &   │──▶│ 03 Operators &             │  │
  │  │    Setup      │   │    Data Types    │   │    Expressions             │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────────┘  │
  │                                                          │                  │
  │                                                          ▼                  │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
  │  │ 06 Arrays &  │◀──│ 05 Functions &   │◀──│ 04 Control Flow            │  │
  │  │    Strings   │   │    Scope         │   │                            │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────────┘  │
  │         │                                                                   │
  │         ▼                                                                   │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
  │  │ 07 Pointers  │──▶│ 08 Structs &     │──▶│ 09 Dynamic Memory          │  │
  │  │ Fundamentals │   │    Unions        │   │                            │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────────┘  │
  │                                                          │                  │
  │                                                          ▼                  │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
  │  │ 12 Build &   │◀──│ 11 Preprocessor  │◀──│ 10 File I/O                │  │
  │  │    Debugging │   │    & Headers     │   │                            │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────────┘  │
  │         │                                                                   │
  │         ▼                                                                   │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────────┐  │
  │  │ 13 Project:  │──▶│ 14 Project:      │──▶│ 15 Project:                │  │
  │  │  Calculator  │   │  Number Guessing │   │  Address Book              │  │
  │  └──────────────┘   └──────────────────┘   └────────────────────────────┘  │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘
```

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| 01 | [Environment Setup](01_Environment_Setup.md) | ⭐ | Compiler, toolchain, Hello World |
| 02 | [Variables and Data Types](02_Variables_and_Data_Types.md) | ⭐ | int, float, char, sizeof, type conversion |
| 03 | [Operators and Expressions](03_Operators_and_Expressions.md) | ⭐ | Arithmetic, comparison, logical, precedence |
| 04 | [Control Flow](04_Control_Flow.md) | ⭐ | if/else, switch, for, while, break/continue |
| 05 | [Functions and Scope](05_Functions_and_Scope.md) | ⭐ | Declaration, parameters, return, scope, recursion |
| 06 | [Arrays and Strings](06_Arrays_and_Strings.md) | ⭐ | Fixed arrays, multidimensional, string functions |
| 07 | [Pointers Fundamentals](07_Pointers_Fundamentals.md) | ⭐⭐ | &, *, NULL, arrays and pointers, pass-by-reference |
| 08 | [Structs and Unions](08_Structs_and_Unions.md) | ⭐⭐ | struct, typedef, union, enum, bit fields |
| 09 | [Dynamic Memory](09_Dynamic_Memory.md) | ⭐⭐ | malloc, calloc, realloc, free, memory leaks |
| 10 | [File I/O](10_File_IO.md) | ⭐⭐ | fopen, fread, fwrite, fprintf, binary vs text |
| 11 | [Preprocessor and Headers](11_Preprocessor_and_Headers.md) | ⭐⭐ | #include, #define, macros, header guards |
| 12 | [Build Tools and Debugging](12_Build_Tools_and_Debugging.md) | ⭐⭐ | Makefile, compiler flags, printf debugging, GDB basics |
| 13 | [Project: Calculator](13_Project_Calculator.md) | ⭐ | scanf, switch-case, functions, input validation |
| 14 | [Project: Number Guessing](14_Project_Number_Guessing.md) | ⭐ | Loops, random numbers, conditionals |
| 15 | [Project: Address Book](15_Project_Address_Book.md) | ⭐⭐ | Structs, arrays, file I/O, CRUD |

## Recommended Learning Order

Follow the lessons sequentially from 01 through 15. Each lesson builds on concepts introduced in the previous one:

1. **Environment Setup (Lesson 1)**: Get your C compiler installed and running
2. **Language Fundamentals (Lessons 2-4)**: Variables, operators, and control flow form the backbone of every C program
3. **Functions (Lesson 5)**: Organize code into reusable, well-scoped functions
4. **Arrays and Strings (Lesson 6)**: Work with fixed-size collections and null-terminated strings
5. **Pointers and Structs (Lessons 7-8)**: Understand memory addresses and build composite data types
6. **Memory and I/O (Lessons 9-10)**: Allocate memory at runtime and interact with the file system
7. **Build Infrastructure (Lessons 11-12)**: Master the preprocessor, Makefiles, and debugging tools
8. **Projects (Lessons 13-15)**: Apply everything you have learned to three increasingly complex projects

## Practice Environment

Verify your C compiler installation:

```bash
gcc --version
# gcc (GCC) 13.x.x (or newer)

# Quick test
echo '#include <stdio.h>
int main(void) { printf("Ready!\n"); return 0; }' > test.c
gcc -Wall -Wextra -std=c11 test.c -o test && ./test
rm -f test test.c
```

Example code for each lesson is available in `examples/C_Basics/`.

## Related Materials

- [C Advanced](../C_Advanced/00_Overview.md) — Systems programming, advanced data structures, and concurrency
- [C++ Basics](../CPP_Basics/00_Overview.md) — C with classes, templates, and the Standard Library
- [Programming](../Programming/00_Overview.md) — Language-independent programming concepts
- [Computer Architecture](../Computer_Architecture/00_Overview.md) — Understanding the hardware that C code runs on
- [Linux](../Linux/00_Overview.md) — The operating system most commonly used for C development

---

**License**: Content licensed under CC BY-NC 4.0
