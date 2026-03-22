# Python Basics

This topic covers Python fundamentals from installation to practical programming patterns. Whether you are writing your first script or solidifying your understanding of core concepts, these lessons will give you a thorough grounding in the language that powers data science, web development, automation, and countless other domains.

## What You'll Learn

This topic provides hands-on coverage of:
- **Getting Started**: Installation, REPL, virtual environments, and tooling
- **Core Language**: Variables, data types, operators, expressions, and control flow
- **Functions**: Defining, calling, and composing functions with Python's rich parameter system
- **Data Structures**: Lists, tuples, dictionaries, sets, and comprehensions
- **Strings**: Formatting, methods, regular expressions, and Unicode
- **Object-Oriented Programming**: Classes, inheritance, polymorphism, dunder methods, and dataclasses
- **Modules and Packages**: Importing, structuring projects, and the standard library
- **File I/O and Exceptions**: Reading/writing files, handling errors, and context managers
- **Pythonic Style**: Idioms, best practices, and patterns that distinguish fluent Python code

## Prerequisites

- [Programming](../Programming/00_Overview.md) — Familiarity with general programming concepts (variables, control flow, functions)

No prior Python experience is required. If you can read pseudocode and understand what a variable or a loop does, you are ready.

## Learning Roadmap

```
                          Python Basics — Learning Path
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 01 Getting    │──▶│ 02 Variables &   │──▶│ 03 Operators &         │  │
  │  │    Started    │   │    Data Types    │   │    Expressions         │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 06 Data      │◀──│ 05 Functions     │◀──│ 04 Control Flow        │  │
  │  │    Structures │   │                  │   │                        │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────┘  │
  │         │                                                               │
  │         ▼                                                               │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 07 Strings   │──▶│ 08 OOP Basics    │──▶│ 09 OOP Advanced        │  │
  │  │              │   │                  │   │                        │  │
  │  └──────────────┘   └──────────────────┘   └────────────┬───────────┘  │
  │                                                          │              │
  │                                                          ▼              │
  │  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
  │  │ 12 Exceptions│◀──│ 11 File I/O      │◀──│ 10 Modules &           │  │
  │  │              │   │                  │   │    Packages            │  │
  │  └──────┬───────┘   └──────────────────┘   └────────────────────────┘  │
  │         │                                                               │
  │         ▼                                                               │
  │  ┌──────────────┐   ┌──────────────────┐                               │
  │  │ 13 Standard  │──▶│ 14 Pythonic      │                               │
  │  │    Library   │   │    Idioms        │                               │
  │  └──────────────┘   └──────────────────┘                               │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

## Lessons

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| 01 | [Getting Started](01_Getting_Started.md) | ⭐ | Installation, REPL, virtual environments, pip, first program |
| 02 | [Variables and Data Types](02_Variables_and_Data_Types.md) | ⭐ | Assignment, int, float, str, bool, None, type conversion |
| 03 | [Operators and Expressions](03_Operators_and_Expressions.md) | ⭐ | Arithmetic, comparison, logical, bitwise, walrus operator |
| 04 | [Control Flow](04_Control_Flow.md) | ⭐ | if/elif/else, for, while, range, match/case, loop patterns |
| 05 | [Functions](05_Functions.md) | ⭐ | Defining, parameters, return values, scope, lambda, closures |
| 06 | [Data Structures](06_Data_Structures.md) | ⭐ | Lists, tuples, dicts, sets, comprehensions, unpacking |
| 07 | [Strings and Text Processing](07_Strings_and_Text_Processing.md) | ⭐ | Methods, formatting, f-strings, regex basics, Unicode |
| 08 | [OOP Basics](08_OOP_Basics.md) | ⭐⭐ | Classes, instances, attributes, methods, __init__, properties |
| 09 | [OOP Advanced](09_OOP_Advanced.md) | ⭐⭐ | Inheritance, polymorphism, dunder methods, dataclasses, ABCs |
| 10 | [Modules and Packages](10_Modules_and_Packages.md) | ⭐ | import, __name__, packages, namespace, third-party libraries |
| 11 | [File I/O](11_File_IO.md) | ⭐ | open(), read/write, context managers, CSV, JSON, pathlib |
| 12 | [Exception Handling](12_Exception_Handling.md) | ⭐⭐ | try/except/finally, custom exceptions, exception chaining |
| 13 | [Standard Library Essentials](13_Standard_Library_Essentials.md) | ⭐ | collections, itertools, functools, datetime, os, sys, re |
| 14 | [Python Idioms and Best Practices](14_Python_Idioms_and_Best_Practices.md) | ⭐⭐ | PEP 8, EAFP vs LBYL, generators, decorators, context managers |

## Recommended Learning Order

Follow the lessons sequentially from 01 through 14. Each lesson builds on concepts introduced in the previous one:

1. **Environment Setup (Lesson 1)**: Get Python installed and running
2. **Language Fundamentals (Lessons 2-4)**: Variables, operators, and control flow form the backbone of every program
3. **Functions (Lesson 5)**: Learn to organize code into reusable blocks
4. **Data Structures and Strings (Lessons 6-7)**: Master Python's built-in collections and text processing
5. **Object-Oriented Programming (Lessons 8-9)**: Model real-world concepts with classes
6. **Project Organization (Lesson 10)**: Structure code into modules and packages
7. **I/O and Error Handling (Lessons 11-12)**: Interact with the file system and handle failures gracefully
8. **Mastery (Lessons 13-14)**: Leverage the standard library and write idiomatic Python

## Practice Environment

Verify your Python installation:

```bash
python3 --version
# Python 3.12.x (or newer)
```

Set up a dedicated virtual environment for exercises:

```bash
# Create a virtual environment
python3 -m venv ~/.venvs/python-basics

# Activate it
source ~/.venvs/python-basics/bin/activate   # macOS / Linux
# .\.venvs\python-basics\Scripts\activate    # Windows

# Confirm
which python
python --version
```

Example code for each lesson is available in `examples/Python_Basics/`.

## Related Materials

- [Python (Advanced)](../Python_Advanced/00_Overview.md) — Decorators, generators, metaclasses, async/await, and performance tuning
- [Programming](../Programming/00_Overview.md) — Language-independent programming concepts
- [Data Science](../Data_Science/00_Overview.md) — Applying Python to data analysis and visualization
- [Testing and QA](../Testing_and_QA/00_Overview.md) — pytest, test-driven development, and quality assurance

---

**License**: Content licensed under CC BY-NC 4.0
