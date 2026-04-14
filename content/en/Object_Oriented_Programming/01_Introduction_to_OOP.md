# Lesson 01: Introduction to OOP

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain what object-oriented programming is and why it matters
2. Compare procedural and object-oriented approaches to problem-solving
3. Trace the historical evolution of OOP from Simula to modern languages
4. Identify the four pillars of OOP at a high level
5. Describe the mental model of "objects as real-world entities"
6. Recognize when OOP is the right paradigm for a given problem
7. Write your first simple class in Python

## What Is Object-Oriented Programming?

Object-Oriented Programming (OOP) is a **programming paradigm** that organizes software around **objects** — bundles of related data and behavior — rather than functions and sequential logic. Instead of writing a list of instructions for the computer to follow, OOP asks: "What are the *things* in my system, and how do they interact?"

```
┌─────────────────────────────────────────────────────┐
│                  Real World                         │
│                                                     │
│   🚗 Car        📱 Phone       🏦 Bank Account     │
│   - color       - brand        - owner              │
│   - speed       - battery      - balance            │
│   - drive()     - call()       - deposit()          │
│   - brake()     - text()       - withdraw()         │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                  OOP Model                          │
│                                                     │
│   class Car:     class Phone:   class BankAccount:  │
│     color         brand           owner             │
│     speed         battery         balance           │
│     drive()       call()          deposit()         │
│     brake()       text()          withdraw()        │
└─────────────────────────────────────────────────────┘
```

The key insight is that OOP mirrors how humans naturally think about the world: as a collection of distinct entities, each with properties (data) and capabilities (behavior).

## Procedural vs Object-Oriented

### Procedural Approach

In procedural programming, you organize code around **procedures** (functions) that operate on data. Data and functions are separate.

```python
# Procedural approach to a bank account
account_owner = "Alice"
account_balance = 1000.0

def deposit(balance, amount):
    if amount <= 0:
        raise ValueError("Amount must be positive")
    return balance + amount

def withdraw(balance, amount):
    if amount > balance:
        raise ValueError("Insufficient funds")
    return balance - amount

# Usage: data is passed around explicitly
account_balance = deposit(account_balance, 500)
account_balance = withdraw(account_balance, 200)
```

Problems with this approach:
- **Data and functions are disconnected**: Nothing ties `account_balance` to `deposit()`
- **No protection**: Any code can modify `account_balance` directly
- **Scaling issues**: With hundreds of accounts, managing separate variables becomes unwieldy
- **No structure**: Relationships between data elements are implicit

### Object-Oriented Approach

In OOP, data and the functions that operate on it are **bundled together** into objects.

```python
# Object-oriented approach
class BankAccount:
    def __init__(self, owner, balance=0.0):
        self.owner = owner
        self._balance = balance  # underscore = "please don't touch directly"

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount

    def withdraw(self, amount):
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount

    @property
    def balance(self):
        return self._balance

# Usage: data and behavior live together
alice = BankAccount("Alice", 1000.0)
alice.deposit(500)
alice.withdraw(200)
print(alice.balance)  # 1300.0
```

### Comparison Summary

```
┌─────────────────────┬──────────────────────┐
│    Procedural       │    Object-Oriented   │
├─────────────────────┼──────────────────────┤
│ Functions + Data    │ Objects (data +      │
│ are separate        │ behavior together)   │
├─────────────────────┼──────────────────────┤
│ Top-down execution  │ Objects interact     │
│ flow                │ via messages         │
├─────────────────────┼──────────────────────┤
│ Global/shared state │ Encapsulated state   │
│ is common           │ per object           │
├─────────────────────┼──────────────────────┤
│ Good for scripts,   │ Good for complex     │
│ small programs      │ systems, GUIs, games │
├─────────────────────┼──────────────────────┤
│ C, Bash, early      │ Python, Java, C++,   │
│ FORTRAN, Pascal     │ C#, Ruby, Swift      │
└─────────────────────┴──────────────────────┘
```

## A Brief History of OOP

Understanding OOP's history helps explain why it looks the way it does.

### Timeline

```
1960s ──── 1970s ──── 1980s ──── 1990s ──── 2000s ──── 2010s+
  │          │          │          │          │          │
Simula    Smalltalk   C++       Java       C#       Rust/Go
(1967)    (1972)     (1979)    (1995)     (2000)   (traits,
 │         │          │         │          │       interfaces)
 │         │          │         │          │
 ▼         ▼          ▼         ▼          ▼
First     Pure OOP   OOP +    OOP for    Modern
classes   "everything procedural enterprise OOP +
& objects  is object" hybrid             other
                                          paradigms
```

- **1967 — Simula** (Ole-Johan Dahl & Kristen Nygaard): Introduced classes, objects, inheritance, and virtual methods for simulation problems. The "grandfather" of OOP.
- **1972 — Smalltalk** (Alan Kay, Xerox PARC): The purest OOP language — everything is an object, even numbers and booleans. Introduced message passing as the core communication mechanism.
- **1979 — C++** (Bjarne Stroustrup): Brought OOP to systems programming by adding classes to C. Proved that OOP could coexist with procedural and low-level code.
- **1995 — Java** (James Gosling, Sun Microsystems): Made OOP mainstream for enterprise. "Write once, run anywhere" with mandatory class-based structure.
- **2000s — C#, Ruby, Python**: Refined OOP with cleaner syntax, dynamic typing, and multi-paradigm support.
- **2010s+ — Modern languages**: Rust (traits), Go (interfaces without inheritance), Kotlin (data classes). OOP ideas are blending with functional programming.

### Alan Kay's Original Vision

Alan Kay, who coined the term "object-oriented programming," described it this way:

> "I thought of objects being like biological cells... able to communicate with messages."

His vision was about **message passing** between autonomous objects, not about inheritance hierarchies. Modern OOP has evolved beyond his original concept, but the core idea of independent, communicating entities remains powerful.

## The Four Pillars of OOP

OOP is built on four foundational principles. We will explore each in depth in subsequent lessons, but here is the overview:

```
┌─────────────────────────────────────────────────┐
│              FOUR PILLARS OF OOP                │
├────────────┬───────────┬───────────┬────────────┤
│            │           │           │            │
│ ENCAPSU-   │ INHERI-   │ POLYMOR-  │ ABSTRAC-   │
│ LATION     │ TANCE     │ PHISM     │ TION       │
│            │           │           │            │
│ Bundling   │ Creating  │ Same      │ Hiding     │
│ data +     │ new       │ interface │ complexity │
│ methods,   │ classes   │ different │ behind     │
│ hiding     │ from      │ behavior  │ simple     │
│ internals  │ existing  │           │ interfaces │
│            │ ones      │           │            │
│ Lesson 04  │ Lesson 05 │ Lesson 07 │ Lesson 08  │
└────────────┴───────────┴───────────┴────────────┘
```

1. **Encapsulation**: Bundle data and the methods that operate on that data together, while restricting direct access to some components. Think of a capsule: the medicine is inside, and you interact through a defined interface.

2. **Inheritance**: Create new classes based on existing ones, inheriting their attributes and methods. A `Dog` class can inherit from `Animal`, gaining `eat()` and `sleep()` while adding `bark()`.

3. **Polymorphism**: Objects of different types can be used through the same interface. A `draw()` method works on circles, squares, and triangles — each drawing itself differently.

4. **Abstraction**: Hide complex implementation details behind simple interfaces. You drive a car without knowing how the engine works internally.

## Why OOP?

### Benefits

- **Modularity**: Code is organized into self-contained objects that can be developed and tested independently
- **Reusability**: Classes can be reused across projects; inheritance allows extending existing code
- **Maintainability**: Changes to one object don't ripple through the entire codebase
- **Modeling**: OOP naturally maps to real-world entities, making designs intuitive
- **Collaboration**: Teams can work on different classes simultaneously with clear interfaces

### When OOP Shines

- Complex systems with many interacting entities (e.g., games, GUIs, enterprise apps)
- Systems that need to model real-world relationships
- Codebases that will grow and evolve over time
- Projects requiring code reuse across teams

### When OOP Is Overkill

- Simple scripts and one-off data transformations
- Pure mathematical computations (functional programming may be better)
- Performance-critical inner loops (OOP overhead may matter)
- Tiny programs under 100 lines

## Your First Class

Let's write a complete class to solidify the concept:

```python
class Dog:
    """A simple Dog class demonstrating OOP basics."""

    # Class attribute: shared by all instances
    species = "Canis familiaris"

    def __init__(self, name, age, breed):
        """Initialize a new Dog instance."""
        self.name = name      # Instance attribute
        self.age = age         # Instance attribute
        self.breed = breed     # Instance attribute

    def bark(self):
        """The dog barks."""
        return f"{self.name} says: Woof!"

    def describe(self):
        """Return a description of the dog."""
        return f"{self.name} is a {self.age}-year-old {self.breed}"

    def birthday(self):
        """Celebrate the dog's birthday."""
        self.age += 1
        return f"Happy birthday, {self.name}! Now {self.age} years old."


# Creating instances (objects)
rex = Dog("Rex", 3, "German Shepherd")
bella = Dog("Bella", 5, "Golden Retriever")

# Using methods
print(rex.bark())        # Rex says: Woof!
print(bella.describe())  # Bella is a 5-year-old Golden Retriever
print(rex.birthday())    # Happy birthday, Rex! Now 4 years old.

# Class attribute is shared
print(rex.species)       # Canis familiaris
print(bella.species)     # Canis familiaris
```

### Anatomy of a Class

```
class Dog:                          <-- Class definition
    species = "Canis familiaris"    <-- Class attribute (shared)

    def __init__(self, name, age):  <-- Constructor (initializer)
        self.name = name            <-- Instance attribute
        self.age = age              <-- Instance attribute

    def bark(self):                 <-- Instance method
        return f"{self.name}: Woof!"

rex = Dog("Rex", 3)                <-- Instantiation (creating object)
rex.bark()                          <-- Method call (sending message)
```

## Key Terminology

| Term | Definition |
|------|-----------|
| **Class** | A blueprint/template for creating objects |
| **Object** | An instance of a class; a concrete entity |
| **Instance** | Synonym for object; emphasizes "created from a class" |
| **Attribute** | Data stored in an object (also called "field" or "property") |
| **Method** | A function defined inside a class that operates on instances |
| **Constructor** | Special method (`__init__` in Python) called when creating an object |
| **Instantiation** | The process of creating an object from a class |
| **Message passing** | Calling a method on an object (sending it a "message") |
| **State** | The current values of an object's attributes |
| **Behavior** | What an object can do, defined by its methods |

## Summary

- OOP organizes code around **objects** that combine data and behavior
- Compared to procedural programming, OOP provides better modularity, encapsulation, and code reuse
- OOP evolved from Simula (1967) through Smalltalk, C++, Java, and into modern multi-paradigm languages
- The four pillars — encapsulation, inheritance, polymorphism, abstraction — form the foundation
- OOP is ideal for complex, evolving systems but can be overkill for simple scripts
- In Python, you define a class with `class`, initialize with `__init__`, and create objects by calling the class

## Next Steps

In [Lesson 02: Classes and Objects](02_Classes_and_Objects.md), we will dive deeper into how classes and objects work, exploring attributes, methods, class vs instance members, and the object lifecycle.
