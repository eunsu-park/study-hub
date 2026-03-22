# C++ Basics

This topic covers fundamental C++ programming from environment setup through classes, the Standard Template Library, and build tools, culminating in a hands-on capstone project. Whether you are transitioning from C or picking up your first systems-level language with object-oriented features, these lessons will give you a solid understanding of modern C++ (C++17) and the idioms that professional codebases rely on every day.

## What You'll Learn

This topic provides hands-on coverage of:

- **Getting Started**: Compiler installation, IDE configuration, and your first compiled C++ program
- **Core Language**: Variables, data types, operators, expressions, and control flow
- **Functions**: Overloading, default parameters, inline functions, and recursion
- **Arrays and Strings**: C-style arrays, `std::string`, `std::string_view`, and string streams
- **Pointers and References**: Address-of and dereference operators, `new`/`delete`, smart pointer preview
- **Namespaces and I/O**: Namespace organization, `iostream`, `iomanip`, and `stringstream`
- **OOP Fundamentals**: Classes, constructors, destructors, encapsulation, operator overloading, and the Rule of Three
- **Inheritance and Polymorphism**: Virtual functions, abstract classes, and multiple inheritance
- **STL**: Containers (`vector`, `map`, `set`), algorithms (`sort`, `find`, `transform`), and iterators
- **Error Handling and File I/O**: Exception hierarchy, `try`/`catch`, `fstream`
- **Build Tools**: CMake basics, targets, and multi-file project organization
- **Project**: A capstone Student Management System that integrates all concepts

## Prerequisites

- [C Basics](../C_Basics/00_Overview.md) -- Familiarity with C fundamentals (variables, control flow, functions, pointers, structs)

If you understand how C programs compile and run and are comfortable with pointers and memory, you are ready.

## Learning Roadmap

```
                         C++ Basics -- Learning Path
  +-----------------------------------------------------------------------------+
  |                                                                             |
  |  +--------------+   +------------------+   +----------------------------+  |
  |  | 01 Environment|-->| 02 Variables &   |-->| 03 Operators &             |  |
  |  |    Setup      |   |    Types         |   |    Control Flow            |  |
  |  +--------------+   +------------------+   +-------------+--------------+  |
  |                                                          |                  |
  |                                                          v                  |
  |  +--------------+   +------------------+   +----------------------------+  |
  |  | 06 Pointers &|<--| 05 Arrays &      |<--| 04 Functions               |  |
  |  |  References  |   |    Strings       |   |                            |  |
  |  +------+-------+   +------------------+   +----------------------------+  |
  |         |                                                                   |
  |         v                                                                   |
  |  +--------------+   +------------------+   +----------------------------+  |
  |  | 07 Namespaces|-->| 08 Classes       |-->| 09 Classes                 |  |
  |  | & IO Streams |   |    Basics        |   |    Advanced                |  |
  |  +--------------+   +------------------+   +-------------+--------------+  |
  |                                                          |                  |
  |                                                          v                  |
  |  +--------------+   +------------------+   +----------------------------+  |
  |  | 12 STL Algo  |<--| 11 STL           |<--| 10 Inheritance &           |  |
  |  | & Iterators  |   |    Containers    |   |    Polymorphism            |  |
  |  +------+-------+   +------------------+   +----------------------------+  |
  |         |                                                                   |
  |         v                                                                   |
  |  +--------------+   +------------------+   +----------------------------+  |
  |  | 13 Exceptions|-->| 14 CMake &       |-->| 15 Project: Student        |  |
  |  | & File IO    |   |    Build Basics  |   |    Management              |  |
  |  +--------------+   +------------------+   +----------------------------+  |
  |                                                                             |
  +-----------------------------------------------------------------------------+
```

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| 01 | [Environment Setup](01_Environment_Setup.md) | ⭐ | Compiler, IDE, Hello World |
| 02 | [Variables and Types](02_Variables_and_Types.md) | ⭐ | int, double, char, bool, auto, const |
| 03 | [Operators and Control Flow](03_Operators_and_Control_Flow.md) | ⭐ | Operators, if/switch, loops |
| 04 | [Functions](04_Functions.md) | ⭐⭐ | Overloading, default params, inline |
| 05 | [Arrays and Strings](05_Arrays_and_Strings.md) | ⭐⭐ | C arrays, std::string, string_view |
| 06 | [Pointers and References](06_Pointers_and_References.md) | ⭐⭐ | Pointers, references, new/delete |
| 07 | [Namespaces and IO Streams](07_Namespaces_and_IO_Streams.md) | ⭐ | Namespace, cin/cout, iomanip, stringstream |
| 08 | [Classes Basics](08_Classes_Basics.md) | ⭐⭐ | Class, constructor, destructor, encapsulation |
| 09 | [Classes Advanced](09_Classes_Advanced.md) | ⭐⭐⭐ | Operator overloading, copy semantics, Rule of Three |
| 10 | [Inheritance and Polymorphism](10_Inheritance_and_Polymorphism.md) | ⭐⭐⭐ | Virtual, abstract, multiple inheritance |
| 11 | [STL Containers](11_STL_Containers.md) | ⭐⭐⭐ | vector, map, set, unordered containers |
| 12 | [STL Algorithms and Iterators](12_STL_Algorithms_and_Iterators.md) | ⭐⭐⭐ | sort, find, transform, iterator categories |
| 13 | [Exceptions and File IO](13_Exceptions_and_File_IO.md) | ⭐⭐⭐ | try/catch, fstream, exception hierarchy |
| 14 | [CMake and Build Basics](14_CMake_and_Build_Basics.md) | ⭐⭐ | CMakeLists.txt, targets, building |
| 15 | [Project: Student Management](15_Project_Student_Management.md) | ⭐⭐⭐ | Capstone project |

## Recommended Learning Order

Follow the lessons sequentially from 01 through 15. Each lesson builds on concepts introduced in the previous one:

1. **Environment Setup (Lesson 1)**: Get your C++ compiler installed and running
2. **Language Fundamentals (Lessons 2-3)**: Variables, operators, and control flow form the backbone of every C++ program
3. **Functions (Lesson 4)**: Overloading, default parameters, and recursion
4. **Data Handling (Lessons 5-6)**: Arrays, strings, pointers, and references
5. **I/O and Organization (Lesson 7)**: Namespaces and formatted input/output
6. **Object-Oriented Programming (Lessons 8-10)**: Classes, inheritance, and polymorphism
7. **Standard Library (Lessons 11-12)**: Containers, algorithms, and iterators
8. **Robustness (Lessons 13-14)**: Exception handling, file I/O, and build systems
9. **Capstone (Lesson 15)**: Apply everything to a Student Management System

## Practice Environment

Verify your C++ compiler installation:

```bash
g++ --version
# g++ (GCC) 13.x.x (or newer)

# Quick test
echo '#include <iostream>
int main() { std::cout << "Ready!\n"; return 0; }' > test.cpp
g++ -std=c++17 -Wall -Wextra test.cpp -o test && ./test
rm -f test test.cpp
```

Example code for each lesson is available in `examples/CPP_Basics/`.

## Related Materials

- [C++ Advanced](../CPP_Advanced/00_Overview.md) -- Templates, modern C++ standards, concurrency, and design patterns
- [C Basics](../C_Basics/00_Overview.md) -- The C language that C++ extends
- [Programming](../Programming/00_Overview.md) -- Language-independent programming concepts
- [Computer Architecture](../Computer_Architecture/00_Overview.md) -- Understanding the hardware that C++ code runs on
- [Software Engineering](../Software_Engineering/00_Overview.md) -- Design patterns and engineering practices

---

**License**: Content licensed under CC BY-NC 4.0
