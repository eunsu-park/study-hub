# C++ Learning Guide

## Introduction

This folder contains materials for systematically learning C++ programming from the ground up. You can progress step by step from basic syntax to modern C++.

**Target Audience**: Beginners to intermediate programmers

---

## Learning Roadmap

```
[Beginner]          [Basics]            [Intermediate]      [Advanced]
  │                   │                   │                   │
  ▼                   ▼                   ▼                   ▼
Environment ───▶ Functions ───────▶ Classes Basics ──▶ Exception Handling
  │                   │                   │                   │
  ▼                   ▼                   ▼                   ▼
Variables ──────▶ Arrays/Strings ──▶ Classes Advanced ▶ Smart Pointers
  │                   │                   │                   │
  ▼                   ▼                   ▼                   ▼
Control Flow ───▶ Pointers/Refs ───▶ Inheritance ──────▶ Modern C++
                                          │
                                          ▼
                                    STL ─────▶ Templates
```

---

## Prerequisites

- Basic computer usage
- Terminal/command prompt experience (recommended)

---

## File List

| Filename | Difficulty | Topics |
|----------|-----------|--------|
| [01_Environment_Setup.md](./01_Environment_Setup.md) | ⭐ | Development environment, Hello World |
| [02_Variables_and_Types.md](./02_Variables_and_Types.md) | ⭐ | Basic types, constants, type casting |
| [03_Operators_and_Control_Flow.md](./03_Operators_and_Control_Flow.md) | ⭐ | Operators, if/switch, loops |
| [04_Functions.md](./04_Functions.md) | ⭐⭐ | Function definition, overloading, default values |
| [05_Arrays_and_Strings.md](./05_Arrays_and_Strings.md) | ⭐⭐ | Arrays, C strings, string class |
| [06_Pointers_and_References.md](./06_Pointers_and_References.md) | ⭐⭐ | Pointers, references, dynamic memory |
| [07_Classes_Basics.md](./07_Classes_Basics.md) | ⭐⭐⭐ | Classes, constructors, destructors |
| [08_Classes_Advanced.md](./08_Classes_Advanced.md) | ⭐⭐⭐ | Operator overloading, copy/move |
| [09_Inheritance_and_Polymorphism.md](./09_Inheritance_and_Polymorphism.md) | ⭐⭐⭐ | Inheritance, virtual functions, abstract classes |
| [10_STL_Containers.md](./10_STL_Containers.md) | ⭐⭐⭐ | vector, map, set |
| [11_STL_Algorithms_Iterators.md](./11_STL_Algorithms_Iterators.md) | ⭐⭐⭐ | algorithm, iterator |
| [12_Templates.md](./12_Templates.md) | ⭐⭐⭐ | Function/class templates |
| [13_Exceptions_and_File_IO.md](./13_Exceptions_and_File_IO.md) | ⭐⭐⭐⭐ | try/catch, fstream |
| [14_Smart_Pointers_Memory.md](./14_Smart_Pointers_Memory.md) | ⭐⭐⭐⭐ | unique_ptr, shared_ptr |
| [15_Modern_CPP.md](./15_Modern_CPP.md) | ⭐⭐⭐⭐ | C++11/14/17/20 features |
| [16_Multithreading_Concurrency.md](./16_Multithreading_Concurrency.md) | ⭐⭐⭐⭐ | std::thread, mutex, async/future |
| [17_CPP20_Advanced.md](./17_CPP20_Advanced.md) | ⭐⭐⭐⭐⭐ | Concepts, Ranges, Coroutines |
| [18_Design_Patterns.md](./18_Design_Patterns.md) | ⭐⭐⭐⭐ | Singleton, Factory, Observer, CRTP |
| [19_Project_Student_Management.md](./19_Project_Student_Management.md) | ⭐⭐⭐ | STL, File I/O, Smart Pointers, Exception Handling |
| [20_CMake_and_Build_Systems.md](./20_CMake_and_Build_Systems.md) | ⭐⭐⭐ | CMakeLists.txt, targets, FetchContent, CTest |
| [21_CPP23_Features.md](./21_CPP23_Features.md) | ⭐⭐⭐⭐⭐ | C++23 new features |
| [22_External_Libraries.md](./22_External_Libraries.md) | ⭐⭐⭐ | Third-party libraries, package managers |

---

## Recommended Learning Path

### Beginner (First Steps in Programming)
1. Environment Setup → Variables and Types → Operators and Control Flow

### Basics (Core Syntax)
2. Functions → Arrays and Strings → Pointers and References

### Intermediate (OOP/STL)
3. Classes Basics → Classes Advanced → Inheritance and Polymorphism
4. STL Containers → STL Algorithms and Iterators → Templates

### Advanced (Expert Level)
5. Exception Handling and File I/O → Smart Pointers and Memory → Modern C++

### In-Depth (Expert)
6. Multithreading and Concurrency → C++20 Advanced → Design Patterns

### Projects
7. Student Management System (comprehensive STL + File I/O project)

---

## Practice Environment

```bash
# Check compiler version
g++ --version

# Compile with C++17 standard
g++ -std=c++17 -Wall -Wextra program.cpp -o program

# Run
./program
```

### Recommended Tools
- **Compiler**: g++ (GCC), clang++
- **IDE**: VS Code + C/C++ extension, CLion, Visual Studio
- **Build System**: CMake (for large projects)

---

## Related Materials

- [C_Programming/](../C_Programming/00_Overview.md) - C language basics
- [Linux/](../Linux/00_Overview.md) - Linux development environment
- [Python/](../Python/00_Overview.md) - Comparison with other languages
