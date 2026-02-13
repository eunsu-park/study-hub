# Shell Scripting Study Guide

## Introduction

This folder provides a systematic study of shell scripting as a programming discipline. Using Bash as the primary shell, it covers advanced techniques, real-world automation patterns, and professional best practices that go well beyond basic scripting.

**Target audience**: Learners who have completed the Linux topic (especially Lesson 09: Shell Scripting basics)

---

## Learning Roadmap

```
[Foundation]              [Intermediate]             [Advanced]
    |                         |                          |
    v                         v                          v
Shell Basics/Env ------> Functions/Libs ---------> Portability/Best Practices
    |                         |                          |
    v                         v                          v
Parameter Expansion ----> I/O & Redirection ------> Testing
    |                         |
    v                         v                     [Projects]
Arrays & Data ----------> String/Regex ----------> Task Runner
    |                         |                          |
    v                         v                          v
Adv. Control Flow ------> Process/Error ----------> Deployment
                              |                          |
                              v                          v
                         Arg Parsing/CLI ---------> Monitoring Tool
```

---

## Prerequisites

- Linux basics and terminal familiarity
- [Linux/09_Shell_Scripting.md](../Linux/09_Shell_Scripting.md) - variables, conditionals, loops, functions, arrays, debugging basics
- [Linux/04_Text_Processing.md](../Linux/04_Text_Processing.md) - grep, sed, awk basics

---

## File List

### Foundation (Review + Beyond Basics)

| File | Difficulty | Key Topics |
|------|-----------|------------|
| [01_Shell_Fundamentals.md](./01_Shell_Fundamentals.md) | ⭐ | Shell types (bash/sh/zsh/dash), POSIX, login/non-login, profile/bashrc loading, exit codes |
| [02_Parameter_Expansion.md](./02_Parameter_Expansion.md) | ⭐⭐ | String manipulation, ${var#}, ${var//}, substring, indirect refs, declare |
| [03_Arrays_and_Data.md](./03_Arrays_and_Data.md) | ⭐⭐ | Associative arrays, stack/queue simulation, CSV parsing, config loading |
| [04_Advanced_Control_Flow.md](./04_Advanced_Control_Flow.md) | ⭐⭐ | [[ ]] vs [ ] vs (( )), extglob, select menus, arithmetic with bc |

### Intermediate (Deep Techniques)

| File | Difficulty | Key Topics |
|------|-----------|------------|
| [05_Functions_and_Libraries.md](./05_Functions_and_Libraries.md) | ⭐⭐ | Return patterns, recursion, function libraries, namespacing, callbacks |
| [06_IO_and_Redirection.md](./06_IO_and_Redirection.md) | ⭐⭐⭐ | File descriptors, here documents, process substitution, named pipes, pipe pitfalls |
| [07_String_Processing.md](./07_String_Processing.md) | ⭐⭐⭐ | Built-in string ops, printf, tr/cut/paste/join, jq/yq for JSON/YAML |
| [08_Regex_in_Bash.md](./08_Regex_in_Bash.md) | ⭐⭐⭐ | =~ operator, BASH_REMATCH, extended regex, glob vs regex, practical validation |
| [09_Process_Management.md](./09_Process_Management.md) | ⭐⭐⭐ | Background jobs, subshells, signals & trap, cleanup patterns, coproc |
| [10_Error_Handling.md](./10_Error_Handling.md) | ⭐⭐⭐ | set -euo pipefail deep dive, trap ERR, error frameworks, ShellCheck, logging |
| [11_Argument_Parsing.md](./11_Argument_Parsing.md) | ⭐⭐⭐ | getopts, getopt, self-documenting help, color output, progress bars |

### Advanced (Professional Techniques)

| File | Difficulty | Key Topics |
|------|-----------|------------|
| [12_Portability_and_Best_Practices.md](./12_Portability_and_Best_Practices.md) | ⭐⭐⭐⭐ | POSIX vs bash vs zsh, bashisms, Google Shell Style Guide, security, performance |
| [13_Testing.md](./13_Testing.md) | ⭐⭐⭐⭐ | Bats framework, unit testing patterns, mocking, TDD, CI integration |

### Projects (Real-World Applications)

| File | Difficulty | Key Topics |
|------|-----------|------------|
| [14_Project_Task_Runner.md](./14_Project_Task_Runner.md) | ⭐⭐⭐ | Makefile-like task runner, dependency management, parallel execution |
| [15_Project_Deployment.md](./15_Project_Deployment.md) | ⭐⭐⭐⭐ | SSH deployment, rolling deploys, Docker entrypoints, rollback |
| [16_Project_Monitor.md](./16_Project_Monitor.md) | ⭐⭐⭐⭐ | Real-time dashboard, alerting, log aggregation, cron integration |

---

## Recommended Learning Order

1. **Foundation (Week 1)**: 01 → 02 → 03 → 04
   - Quickly review shell basics, then dive into parameter expansion and arrays
2. **Intermediate (Week 2-3)**: 05 → 06 → 07 → 08 → 09 → 10 → 11
   - Core scripting techniques: I/O, regex, processes, error handling
3. **Advanced (Week 4)**: 12 → 13
   - Portability, best practices, and testing
4. **Projects (Week 5)**: 14 → 15 → 16
   - Apply everything in real-world projects

---

## Practice Environment

```bash
# Check your bash version (4.0+ recommended for associative arrays)
bash --version

# Install ShellCheck for static analysis
# macOS
brew install shellcheck

# Ubuntu/Debian
sudo apt install shellcheck

# Install Bats for testing (Lesson 13)
brew install bats-core  # macOS
# or from source: https://github.com/bats-core/bats-core
```

---

## Related Topics

- [Linux/](../Linux/00_Overview.md) - Linux fundamentals, shell basics
- [Git/](../Git/00_Overview.md) - Version control (scripts often used in Git hooks)
- [Docker/](../Docker/00_Overview.md) - Container entrypoints use shell scripts
- [MLOps/](../MLOps/00_Overview.md) - Automation pipelines

---

## References

- [GNU Bash Manual](https://www.gnu.org/software/bash/manual/)
- [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
- [ShellCheck Wiki](https://www.shellcheck.net/wiki/)
- [Bats-core Documentation](https://bats-core.readthedocs.io/)
- [Advanced Bash-Scripting Guide](https://tldp.org/LDP/abs/html/)
