# What Is Programming

**Next**: [Programming Paradigms](./02_Programming_Paradigms.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define programming and distinguish it from coding
2. Apply the four components of computational thinking (decomposition, pattern recognition, abstraction, algorithmic thinking) to break down problems
3. Follow the structured five-step problem-solving process used by professional programmers
4. Describe the different levels of abstraction in software, from machine code to high-level frameworks
5. Compare and contrast compilation, interpretation, and JIT compilation as program execution models
6. Outline the phases of the Software Development Lifecycle (SDLC)
7. Explain why code readability and communication are essential aspects of programming
8. Trace the historical evolution of programming languages from the 1840s to the modern era

---

Programming is one of the most broadly applicable skills in modern technology. Whether you are building a web application, automating a scientific experiment, or training a machine learning model, the underlying discipline is the same: translating human intent into precise instructions a computer can execute. This lesson establishes the foundational vocabulary and mental models you will rely on throughout the rest of the course.

## What Is Programming?

Programming is the art and science of instructing computers to perform tasks. At its core, programming is about **problem solving** — taking a problem from the real world, breaking it down into logical steps, and expressing those steps in a language that a computer can execute.

### Programming vs Coding

While the terms are often used interchangeably, there's a subtle but important distinction:

- **Coding**: The act of writing code in a specific programming language — the syntax, the keywords, the mechanics.
- **Programming**: The broader discipline that includes understanding problems, designing solutions, writing code, testing, debugging, and maintaining software over time.

A programmer doesn't just write code; they:
- Analyze problems
- Design algorithms
- Choose appropriate data structures
- Write maintainable, readable code
- Test and debug
- Collaborate with others
- Document their work
- Refactor and improve existing code

> **📌 Analogy:** Coding is to programming what typing is to writing a novel. Typing is a necessary skill, but writing involves creativity, structure, character development, plot, and revision.

---

## Computational Thinking

Computational thinking is a problem-solving methodology that involves breaking down complex problems into manageable parts. It consists of four key components:

### 1. Decomposition

Breaking a large problem into smaller, more manageable sub-problems.

**Example**: Planning a trip
- Large problem: "Plan a vacation to Japan"
- Decomposed:
  - Research destinations
  - Book flights
  - Reserve hotels
  - Plan daily itineraries
  - Budget expenses
  - Learn basic phrases

### 2. Pattern Recognition

Identifying similarities, patterns, or trends that can help solve problems more efficiently.

**Example**: You notice that every login system needs:
- Username/password input
- Validation
- Encryption
- Session management
- Password reset mechanism

Once you've built one, you can recognize this pattern and reuse your solution.

### 3. Abstraction

Focusing on the essential features while ignoring irrelevant details. Abstraction helps us manage complexity.

**Example**: When you drive a car, you use abstractions:
- "Turn the wheel" — you don't think about the mechanical linkage to the tires
- "Press the brake" — you don't consider the hydraulic system
- You interact with the **interface** (steering wheel, pedals) without needing to understand the **implementation** (engine, transmission, brakes)

In programming:
- Using a function `sort(array)` — you don't need to know if it's quicksort, mergesort, or heapsort
- Calling `database.save(user)` — the internal SQL, connection pooling, and transactions are abstracted away

### 4. Algorithmic Thinking

Designing step-by-step instructions (algorithms) to solve a problem. An algorithm must be:
- **Precise**: Each step is clearly defined
- **Unambiguous**: No room for interpretation
- **Finite**: It must eventually terminate
- **General**: It works for a range of inputs, not just one case

**Example**: Algorithm for making tea
```
1. Fill kettle with water
2. Turn on kettle
3. While water is not boiling:
   - Wait
4. Place tea bag in cup
5. Pour boiling water into cup
6. Wait 3-5 minutes
7. Remove tea bag
8. Add milk/sugar if desired
9. Done
```

---

## The Problem-Solving Process

Professional programmers follow a structured process:

### 1. Understand the Problem

- What is the input?
- What is the expected output?
- What are the constraints?
- What are the edge cases?

**Example**: "Write a program to find the largest number in a list"
- Input: A list of numbers (could be empty? negative?)
- Output: The largest number (what if list is empty?)
- Constraints: Time? Memory? Size of list?

### 2. Plan the Solution

- Break down the problem (decomposition)
- Design an algorithm
- Choose data structures
- Consider multiple approaches

**Example approaches for finding largest number**:
- Approach 1: Sort the list, take the last element
- Approach 2: Iterate through list, keeping track of max seen so far
- Which is better? (Approach 2: O(n) vs O(n log n), less memory)

### 3. Implement the Solution

Translate your algorithm into code. Here's the same solution in multiple languages:

**Python:**
```python
def find_largest(numbers):
    if not numbers:
        return None

    largest = numbers[0]
    for num in numbers:
        if num > largest:
            largest = num
    return largest

# Usage
print(find_largest([3, 7, 2, 9, 1]))  # Output: 9
```

**JavaScript:**
```javascript
function findLargest(numbers) {
    if (numbers.length === 0) {
        return null;
    }

    let largest = numbers[0];
    for (let num of numbers) {
        if (num > largest) {
            largest = num;
        }
    }
    return largest;
}

// Usage
console.log(findLargest([3, 7, 2, 9, 1]));  // Output: 9
```

**Java:**
```java
public class Main {
    public static Integer findLargest(int[] numbers) {
        if (numbers.length == 0) {
            return null;
        }

        int largest = numbers[0];
        for (int num : numbers) {
            if (num > largest) {
                largest = num;
            }
        }
        return largest;
    }

    public static void main(String[] args) {
        int[] numbers = {3, 7, 2, 9, 1};
        System.out.println(findLargest(numbers));  // Output: 9
    }
}
```

**C++:**
```cpp
#include <iostream>
#include <vector>
#include <optional>

std::optional<int> findLargest(const std::vector<int>& numbers) {
    if (numbers.empty()) {
        return std::nullopt;
    }

    int largest = numbers[0];
    for (int num : numbers) {
        if (num > largest) {
            largest = num;
        }
    }
    return largest;
}

int main() {
    std::vector<int> numbers = {3, 7, 2, 9, 1};
    auto result = findLargest(numbers);
    if (result) {
        std::cout << *result << std::endl;  // Output: 9
    }
    return 0;
}
```

Notice: The **algorithm is the same** across all languages — only the syntax differs.

### 4. Test the Solution

- Test with normal inputs
- Test edge cases (empty list, single element, negative numbers)
- Test boundary conditions

```python
# Test cases
assert find_largest([3, 7, 2, 9, 1]) == 9
assert find_largest([5]) == 5
assert find_largest([-3, -7, -2]) == -2
assert find_largest([]) is None
```

### 5. Refine and Optimize

- Is the code readable?
- Can it be more efficient?
- Is it maintainable?

**Refinement** (using built-in functions):
```python
def find_largest(numbers):
    return max(numbers) if numbers else None
```

More concise, leverages tested library code, and just as clear.

---

## Levels of Abstraction

Programs exist at multiple levels, from hardware to high-level frameworks:

### 1. Machine Code (Lowest Level)

Binary instructions the CPU executes directly:
```
10110000 01100001  // Move 'a' into register AL
```

### 2. Assembly Language

Human-readable mnemonics for machine instructions:
```assembly
MOV AL, 61h  ; Move 'a' into register AL
```

### 3. High-Level Languages

Abstracted syntax closer to human language:
```python
letter = 'a'
```

### 4. Frameworks and Libraries (Highest Level)

Pre-built solutions for common tasks:
```python
# Django web framework
from django.http import HttpResponse

def hello(request):
    return HttpResponse("Hello, world!")
```

**The Power of Abstraction**: You can build web applications without understanding TCP/IP, HTTP headers, socket programming, or assembly. Each layer abstracts away the complexity of the layer below.

---

## The Role of a Programmer

A programmer is a **problem solver** who happens to use code as a tool. Key responsibilities:

1. **Understanding Requirements**: What problem are we solving? For whom?
2. **Designing Solutions**: Architecture, algorithms, data structures
3. **Writing Code**: Translating designs into working software
4. **Testing**: Ensuring correctness and reliability
5. **Debugging**: Finding and fixing errors
6. **Maintaining**: Updating, refactoring, improving existing code
7. **Collaborating**: Working with other developers, designers, stakeholders
8. **Documenting**: Making code understandable for future developers (including future you)
9. **Learning**: Technologies change rapidly; continuous learning is essential

**Programming is about communication**:
- Communication with the **computer**: precise, unambiguous instructions
- Communication with **other developers**: clear, maintainable, documented code
- Communication with **stakeholders**: understanding needs, setting expectations

---

## How Computers Execute Programs

Understanding execution models helps you reason about performance and behavior.

### Compilation

**Source code** → **Compiler** → **Machine code** → **Execution**

Languages: C, C++, Rust, Go

**Process**:
1. Write source code: `hello.c`
2. Compile: `gcc hello.c -o hello`
3. Run: `./hello`

**Pros**:
- Fast execution (optimized machine code)
- Errors caught at compile time
- No runtime dependency on compiler

**Cons**:
- Slower development cycle (must recompile)
- Platform-specific binaries (must recompile for Windows/Mac/Linux)

### Interpretation

**Source code** → **Interpreter** → **Execution** (line by line)

Languages: Python, Ruby, JavaScript (early versions)

**Process**:
1. Write source code: `hello.py`
2. Run: `python hello.py`

**Pros**:
- Fast development cycle (no compilation step)
- Platform-independent (just need the interpreter)
- Dynamic features (eval, runtime code generation)

**Cons**:
- Slower execution (no optimization)
- Errors found at runtime
- Requires interpreter to be installed

### Just-In-Time (JIT) Compilation

**Source code** → **Bytecode** → **JIT Compiler** → **Machine code** → **Execution**

Languages: Java, C#, JavaScript (modern engines like V8)

**Process** (Java example):
1. Write: `Hello.java`
2. Compile to bytecode: `javac Hello.java` → `Hello.class`
3. Run: `java Hello` (JVM compiles bytecode to machine code on-the-fly)

**Pros**:
- Platform-independent bytecode
- Optimizations at runtime (based on actual usage patterns)
- Faster than pure interpretation

**Cons**:
- Warm-up time (initial execution slower)
- More memory overhead

---

## The Software Development Lifecycle (SDLC)

Writing code is just one phase. Professional software development includes:

1. **Planning**: What are we building? Why?
2. **Analysis**: Requirements gathering, feasibility study
3. **Design**: Architecture, database schema, UI/UX mockups
4. **Implementation**: Writing code
5. **Testing**: Unit tests, integration tests, user acceptance testing
6. **Deployment**: Releasing to production
7. **Maintenance**: Bug fixes, updates, new features

This is often iterative: Agile, Scrum, Kanban methodologies involve repeated cycles of plan → build → test → release.

---

## Brief History of Programming

### Ada Lovelace (1843)
- Wrote the first algorithm intended for machine processing
- Notes on Charles Babbage's Analytical Engine
- Considered the first programmer

### Alan Turing (1936)
- Turing Machine: theoretical foundation of computation
- Turing Test: defining artificial intelligence
- Codebreaker in WWII

### Early Languages (1950s-1960s)
- **FORTRAN** (1957): Formula Translation, for scientific computing
- **COBOL** (1959): Business applications
- **LISP** (1958): Artificial intelligence, functional programming

### Structured Programming (1970s)
- **C** (1972): Systems programming, influenced nearly all modern languages
- **Pascal** (1970): Teaching programming, structured design

### Object-Oriented Era (1980s-1990s)
- **C++** (1985): OOP + systems programming
- **Java** (1995): "Write once, run anywhere"
- **Python** (1991): Readability, simplicity

### Modern Era (2000s-Present)
- **JavaScript**: Evolved from browser scripting to full-stack development
- **Rust** (2010): Memory safety without garbage collection
- **Go** (2009): Simplicity, concurrency, efficiency
- **Kotlin**, **Swift**: Modern, expressive, safe
- Rise of frameworks: React, Angular, Django, Rails

---

## Programming as Communication

Code is read far more often than it is written. You're not just instructing a computer; you're communicating intent to:

- **Future you** (in 6 months, you'll forget why you wrote it this way)
- **Team members** (who need to understand, modify, extend your code)
- **Maintainers** (who will fix bugs and add features)

### Bad Code (Communicates Poorly)

```python
def f(x):
    return [i for i in x if i % 2 == 0]
```

What does `f` do? What is `x`? What does it return?

### Good Code (Communicates Clearly)

```python
def filter_even_numbers(numbers):
    """
    Returns a list of even numbers from the input list.

    Args:
        numbers: List of integers

    Returns:
        List of even integers from the input
    """
    return [num for num in numbers if num % 2 == 0]
```

Now it's immediately clear:
- Purpose: filter even numbers
- Input: list of numbers
- Output: list of even numbers
- Implementation: readable, with meaningful variable names

---

## Exercises

### Exercise 1: Decomposition

Break down the following real-world problem into computational steps:

**Problem**: Build a simple library management system

- What are the main components?
- What data needs to be stored?
- What operations are needed?

### Exercise 2: Abstraction

Describe the following at different levels of abstraction:

**Task**: Send an email

- High-level abstraction (user perspective)
- Mid-level abstraction (application perspective)
- Low-level abstraction (network/protocol perspective)

### Exercise 3: Algorithmic Thinking

Write a step-by-step algorithm (in plain English or pseudocode) for:

**Problem**: Determine if a word is a palindrome (reads the same forwards and backwards)

- Example: "racecar" → true, "hello" → false

### Exercise 4: Problem Solving

Apply the 5-step problem-solving process:

**Problem**: Write a function that counts how many times a specific word appears in a sentence.

1. Understand: What are the inputs? Outputs? Edge cases?
2. Plan: What algorithm will you use?
3. Implement: Write it in at least two different languages
4. Test: Write test cases
5. Refine: Can you make it better?

### Exercise 5: Computational Thinking

Apply the four components of computational thinking to:

**Problem**: Organize a large music collection

- Decomposition: What sub-problems exist?
- Pattern Recognition: What patterns can you identify?
- Abstraction: What essential features matter? What can be ignored?
- Algorithm: Outline steps to organize the collection

### Exercise 6: Code Communication

Refactor this code to be more communicative:

```python
def p(l):
    r = 1
    for i in l:
        r *= i
    return r
```

- Use meaningful names
- Add documentation
- Consider edge cases

---

## Summary

Programming is:
- **Problem solving** using computational thinking
- **Communication** with computers and humans
- **A process**: understand → plan → implement → test → refine
- **Multi-level**: from machine code to high-level frameworks
- **Execution models**: compilation, interpretation, JIT
- **A discipline**: analysis, design, implementation, testing, maintenance

The best programmers are not those who know the most syntax, but those who can **think clearly, solve problems systematically, and write code that communicates intent**.

---

**Next**: [Programming Paradigms](./02_Programming_Paradigms.md)
