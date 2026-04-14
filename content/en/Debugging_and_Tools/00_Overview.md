# Debugging and Tools

## Introduction

This folder contains materials for learning how to debug code, diagnose errors, and use essential developer tools effectively. You will progress from reading error messages through systematic debugging strategies to profiling and automated quality checks.

**Target Audience**: Beginner developers who want to move beyond "it doesn't work" and learn to methodically find and fix bugs.

---

## Learning Roadmap

```
[Foundations]              [Strategy]               [Tooling]
    |                         |                        |
    v                         v                        v
Reading Errors -------> Debugging Strategy ----> Linters & Formatters
    |                         |                        |
    v                         v                        v
Print Debugging -------> Logging ------------> Type Checking
    |                         |                        |
    v                         v                        v
Using a Debugger         Testing Basics ------> Profiling Basics
    |                                                  |
    v                                                  v
Common Bug Patterns      VCS for Debugging ---> Debugging Workflow
```

---

## What You'll Learn

- How to read Python tracebacks and decode error messages
- Strategic use of print statements and the logging module
- Hands-on use of `pdb`, `breakpoint()`, and IDE debuggers
- Recognition of common bug patterns (off-by-one, mutable defaults, scope issues)
- Systematic debugging strategies (binary search, minimal reproduction)
- Writing tests with `assert`, `unittest`, and `pytest`
- Using linters (`pylint`, `flake8`, `ruff`) and formatters (`black`)
- Adding type hints and checking them with `mypy`
- Profiling code with `cProfile`, `timeit`, and memory profilers
- Using `git bisect`, `git blame`, and `git diff` for debugging
- Combining all techniques into a real-world debugging workflow

---

## Prerequisites

- Basic Python programming (variables, functions, loops, classes)
- Basic terminal/command-line usage
- Basic Git knowledge (helpful for Lesson 11)

---

## File List

| Filename | Difficulty | Key Topics |
|----------|------------|------------|
| [01_Reading_Error_Messages.md](./01_Reading_Error_Messages.md) | ⭐ | Traceback anatomy, common error types, reading stack traces |
| [02_Print_Debugging.md](./02_Print_Debugging.md) | ⭐ | Strategic print, f-string tricks, temporary vs permanent debug output |
| [03_Using_a_Debugger.md](./03_Using_a_Debugger.md) | ⭐ | pdb, breakpoint(), stepping, IDE debugger |
| [04_Common_Bug_Patterns.md](./04_Common_Bug_Patterns.md) | ⭐⭐ | Off-by-one, mutable defaults, scope, None handling |
| [05_Debugging_Strategy.md](./05_Debugging_Strategy.md) | ⭐⭐ | Binary search, reproduction, minimal examples, rubber duck |
| [06_Logging.md](./06_Logging.md) | ⭐⭐ | logging module, levels, formatters, handlers, configuration |
| [07_Testing_Basics.md](./07_Testing_Basics.md) | ⭐⭐ | assert, unittest, pytest, test-driven debugging |
| [08_Linters_and_Formatters.md](./08_Linters_and_Formatters.md) | ⭐⭐ | pylint, flake8, black, ruff, pre-commit hooks |
| [09_Type_Checking.md](./09_Type_Checking.md) | ⭐⭐ | Type hints, mypy, gradual typing, common type errors |
| [10_Profiling_Basics.md](./10_Profiling_Basics.md) | ⭐⭐⭐ | cProfile, timeit, memory_profiler, line_profiler |
| [11_Version_Control_for_Debugging.md](./11_Version_Control_for_Debugging.md) | ⭐⭐⭐ | git bisect, git blame, git diff, debugging with VCS |
| [12_Debugging_Workflow.md](./12_Debugging_Workflow.md) | ⭐⭐⭐ | End-to-end case studies, combining all techniques |

---

## Recommended Learning Path

### Stage 1: Error Reading and Basic Debugging
1. Reading Error Messages -> Print Debugging -> Using a Debugger

### Stage 2: Patterns and Strategy
2. Common Bug Patterns -> Debugging Strategy

### Stage 3: Logging and Testing
3. Logging -> Testing Basics

### Stage 4: Automated Tools
4. Linters and Formatters -> Type Checking -> Profiling Basics

### Stage 5: Advanced Workflows
5. Version Control for Debugging -> Debugging Workflow

---

## Quick Start

### Try the Debugger

```python
# Save this as buggy.py and debug it
def average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)

# This will crash -- can you find why?
result = average([])
print(f"Average: {result}")
```

```bash
# Run with debugger
python -m pdb buggy.py
```

---

## Related Materials

- [Programming](../Programming/00_Overview.md) - Programming fundamentals
- [Python Basics](../Python_Basics/00_Overview.md) - Python language basics
- [Testing and QA](../Testing_and_QA/00_Overview.md) - Advanced testing techniques
- [Git](../Git/00_Overview.md) - Version control for debugging
