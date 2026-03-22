# C++ Advanced

## Introduction

This topic covers advanced C++ programming: templates and metaprogramming, modern C++ standards (C++11 through C++23), concurrency and multithreading, design patterns, and build system integration. These lessons build on fundamental C++ knowledge to develop mastery of the language's most powerful features.

**Prerequisites**: [CPP_Basics](../CPP_Basics/00_Overview.md) (or equivalent knowledge of classes, inheritance, STL containers, and basic templates)

---

## Learning Roadmap

```
[Templates & Memory]               [Modern Standards]              [Systems & Patterns]
  |                                   |                               |
  v                                   v                               v
Move Semantics ------+          Modern C++11/14 ---+            Multithreading
  |                  |            |                 |              |
  v                  |            v                 |              v
Templates            |          Modern C++17        |            Concurrency Advanced
  |                  |            |                 |              |
  v                  |            v                 |              v
Template Metaprog    |          C++20 Concepts      |            Design Patterns
  |                  |            |                 |              (Creational/Structural)
  v                  |            v                 |              |
Smart Pointers       |          C++20 Ranges        |              v
  & RAII             |            |                 |            Design Patterns
  |                  |            v                 |              (Behavioral/Idioms)
  v                  |          C++20 Coroutines    |              |
Error Handling ------+            |                 |              v
                                  v                 |            External Libraries
                                Modules & C++20 ----+              & Build
                                  Utilities
                                  |
                                  v
                                C++23 Features
```

---

## File List

| # | Title | Difficulty | Key Content |
|---|-------|-----------|-------------|
| [01](./01_Move_Semantics_Deep_Dive.md) | Move Semantics Deep Dive | ⭐⭐⭐ | rvalue refs, std::move, forwarding, Rule of Five/Zero |
| [02](./02_Templates.md) | Templates | ⭐⭐⭐ | function/class templates, specialization |
| [03](./03_Template_Metaprogramming.md) | Template Metaprogramming | ⭐⭐⭐⭐ | SFINAE, type_traits, if constexpr |
| [04](./04_Smart_Pointers_and_RAII.md) | Smart Pointers and RAII | ⭐⭐⭐⭐ | unique_ptr, shared_ptr, weak_ptr, RAII |
| [05](./05_Error_Handling_Patterns.md) | Error Handling Patterns | ⭐⭐⭐ | noexcept, exception safety, std::expected |
| [06](./06_Modern_CPP_11_14.md) | Modern C++ (C++11/14) | ⭐⭐⭐ | auto, lambda, constexpr, uniform init |
| [07](./07_Modern_CPP_17.md) | Modern C++ (C++17) | ⭐⭐⭐ | structured bindings, optional/variant/any, filesystem |
| [08](./08_CPP20_Concepts.md) | C++20 Concepts | ⭐⭐⭐⭐ | concepts, requires, constrained auto |
| [09](./09_CPP20_Ranges.md) | C++20 Ranges | ⭐⭐⭐⭐ | views, adaptors, pipeline composition |
| [10](./10_CPP20_Coroutines.md) | C++20 Coroutines | ⭐⭐⭐⭐⭐ | co_await, co_yield, generators |
| [11](./11_Modules_and_CPP20_Utilities.md) | Modules and C++20 Utilities | ⭐⭐⭐ | export/import, std::format, std::span |
| [12](./12_Multithreading.md) | Multithreading | ⭐⭐⭐⭐ | std::thread, mutex, async/future |
| [13](./13_Concurrency_Advanced.md) | Concurrency Advanced | ⭐⭐⭐⭐⭐ | latch/barrier, lock-free, memory_order |
| [14](./14_Design_Patterns_Creational_Structural.md) | Design Patterns (Creational/Structural) | ⭐⭐⭐⭐ | SOLID, Singleton, Factory, Adapter, Decorator |
| [15](./15_Design_Patterns_Behavioral_Idioms.md) | Design Patterns (Behavioral/Idioms) | ⭐⭐⭐⭐ | Observer, Strategy, CRTP, PIMPL |
| [16](./16_CPP23_Features.md) | C++23 Features | ⭐⭐⭐⭐⭐ | std::expected, std::print, deducing this |
| [17](./17_External_Libraries_and_Build.md) | External Libraries and Build | ⭐⭐⭐ | Conan, vcpkg, FetchContent, CTest, Boost/fmt |

---

## Recommended Learning Order

### Path 1: Templates & Memory
1. Move Semantics Deep Dive -> Templates -> Template Metaprogramming -> Smart Pointers and RAII -> Error Handling Patterns

### Path 2: Modern Standards
2. Modern C++11/14 -> Modern C++17 -> C++20 Concepts -> C++20 Ranges -> C++20 Coroutines -> Modules and C++20 Utilities -> C++23 Features

### Path 3: Systems & Patterns
3. Multithreading -> Concurrency Advanced -> Design Patterns (Creational/Structural) -> Design Patterns (Behavioral/Idioms) -> External Libraries and Build

---

## Practice Environment

```bash
# Check compiler version (C++20 support required for most lessons)
g++ --version
clang++ --version

# Compile with C++20 and warnings
g++ -std=c++20 -Wall -Wextra -pedantic -g program.cpp -o program

# Compile with C++23 features (GCC 13+ / Clang 17+)
g++ -std=c++23 -Wall -Wextra -pedantic -g program.cpp -o program

# Compile with AddressSanitizer
g++ -std=c++20 -fsanitize=address -g program.cpp -o program

# Compile with ThreadSanitizer (for concurrency lessons)
g++ -std=c++20 -fsanitize=thread -g program.cpp -o program -pthread
```

---

## Related Materials

- [CPP_Basics](../CPP_Basics/00_Overview.md) - C++ fundamentals (variables, OOP, STL, basic templates)
- [C_Advanced/](../C_Advanced/00_Overview.md) - Advanced C programming (pointers, systems programming)
- [Algorithm/](../Algorithm/00_Overview.md) - Data structures and algorithms
- [Software_Engineering/](../Software_Engineering/00_Overview.md) - Software design principles
- [System_Design/](../System_Design/00_Overview.md) - System architecture and design
