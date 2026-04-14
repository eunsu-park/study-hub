# Object-Oriented Programming

A comprehensive guide to object-oriented programming (OOP), the dominant paradigm for structuring complex software systems. This topic covers everything from foundational concepts like classes and objects to advanced principles such as SOLID, design patterns, and modern Python OOP features. While explanations are language-agnostic, all implementations use Python for clarity and accessibility.

## What You'll Learn

This topic provides thorough coverage of:
- **Core Building Blocks**: Classes, objects, constructors, attributes, and methods
- **Four Pillars**: Encapsulation, inheritance, polymorphism, and abstraction
- **Composition Patterns**: Composition vs inheritance, delegation, and mixins
- **Design Principles**: SOLID principles for maintainable, extensible code
- **Design Patterns**: Classic Gang of Four patterns (Singleton, Factory, Observer, Strategy)
- **Python OOP Features**: Magic methods, dataclasses, protocols, and modern idioms
- **Best Practices**: Anti-patterns to avoid, refactoring strategies, and real-world guidelines

## Prerequisites

- **Basic programming knowledge**: Variables, control flow, functions, basic data structures
- **Familiarity with Python syntax**: The [Programming](../Programming/00_Overview.md) topic is recommended

## Lessons

| # | Title | Description |
|---|-------|-------------|
| 01 | [Introduction to OOP](01_Introduction_to_OOP.md) | Procedural vs OOP, history, motivation, mental models |
| 02 | [Classes and Objects](02_Classes_and_Objects.md) | Definitions, instances, attributes, methods |
| 03 | [Constructors and Initialization](03_Constructors_and_Initialization.md) | `__init__`, `self`, default values, validation |
| 04 | [Encapsulation](04_Encapsulation.md) | Access control, getters/setters, `@property` |
| 05 | [Inheritance](05_Inheritance.md) | Basic inheritance, `super()`, method overriding |
| 06 | [Multiple Inheritance](06_Multiple_Inheritance.md) | MRO, diamond problem, mixins |
| 07 | [Polymorphism](07_Polymorphism.md) | Duck typing, operator overloading, protocols |
| 08 | [Abstraction](08_Abstraction.md) | ABC, abstract methods, interfaces |
| 09 | [Composition vs Inheritance](09_Composition_vs_Inheritance.md) | Has-a vs is-a, delegation, when to use which |
| 10 | [SOLID Principles](10_SOLID_Principles.md) | SRP, OCP, LSP, ISP, DIP |
| 11 | [Design Patterns Intro](11_Design_Patterns_Intro.md) | Singleton, Factory, Observer, Strategy |
| 12 | [Magic Methods](12_Magic_Methods.md) | `__str__`, `__repr__`, `__eq__`, `__hash__`, `__iter__` |
| 13 | [Dataclasses and Modern OOP](13_Dataclasses_and_Modern_OOP.md) | `@dataclass`, `NamedTuple`, `Protocol` |
| 14 | [OOP Best Practices](14_OOP_Best_Practices.md) | Anti-patterns, refactoring, practical guidelines |

## Learning Roadmap

```
                    ┌─────────────────────┐
                    │  01 Introduction    │
                    │     to OOP          │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │  02 Classes and     │
                    │     Objects         │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │  03 Constructors    │
                    │  and Initialization │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐ ┌────▼────┐ ┌───────▼───────┐
     │ 04 Encapsu-   │ │ 05 In-  │ │ 07 Polymor-   │
     │    lation     │ │ herit.  │ │    phism       │
     └────────┬──────┘ └────┬────┘ └───────┬───────┘
              │             │              │
              │        ┌────▼────┐         │
              │        │ 06 Mul- │         │
              │        │ tiple   │         │
              │        └────┬────┘         │
              │             │              │
              └──────────┬──┴──────────────┘
                         │
                ┌────────▼────────────┐
                │  08 Abstraction     │
                └────────┬────────────┘
                         │
                ┌────────▼────────────┐
                │  09 Composition vs  │
                │     Inheritance     │
                └────────┬────────────┘
                         │
              ┌──────────┼──────────────┐
              │          │              │
     ┌────────▼──────┐   │    ┌────────▼──────┐
     │ 10 SOLID      │   │    │ 11 Design     │
     │ Principles    │   │    │ Patterns      │
     └────────┬──────┘   │    └────────┬──────┘
              │          │             │
              └──────────┼─────────────┘
                         │
                ┌────────▼────────────┐
                │  12 Magic Methods   │
                └────────┬────────────┘
                         │
                ┌────────▼────────────┐
                │  13 Dataclasses &   │
                │  Modern OOP         │
                └────────┬────────────┘
                         │
                ┌────────▼────────────┐
                │  14 OOP Best        │
                │  Practices          │
                └─────────────────────┘
```
