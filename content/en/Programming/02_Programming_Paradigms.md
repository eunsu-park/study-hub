# Programming Paradigms

**Previous**: [What Is Programming](./01_What_Is_Programming.md) | **Next**: [Data Types & Abstraction](./03_Data_Types_and_Abstraction.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define what a programming paradigm is and explain why different paradigms exist
2. Describe the imperative and procedural programming styles and identify their distinguishing characteristics
3. Explain the core concepts of Object-Oriented Programming (encapsulation, inheritance, polymorphism, abstraction)
4. Contrast functional programming with imperative programming, focusing on pure functions and immutability
5. Distinguish declarative programming from imperative programming with concrete examples (SQL, HTML)
6. Recognize logic programming, event-driven programming, and reactive programming as distinct paradigms
7. Evaluate trade-offs between paradigms and select the most appropriate one for a given problem domain
8. Demonstrate how modern multi-paradigm languages blend multiple styles in a single program

---

Every programming language embodies one or more paradigms -- fundamental approaches to organizing and reasoning about code. Choosing the right paradigm for a problem is as important as choosing the right algorithm, because it shapes how you decompose the problem, manage state, and compose your solution. This lesson surveys the major paradigms and equips you to make informed choices.

## What Is a Programming Paradigm?

A **programming paradigm** is a fundamental style or approach to programming. It defines:
- How you structure your code
- How you think about problems
- What concepts and abstractions you use

Think of paradigms as different **philosophies** for solving problems with code.

> **📌 Analogy:** Just as there are different architectural styles (Gothic, Modernist, Brutalist), there are different programming styles. Each has its strengths, trade-offs, and appropriate use cases.

---

## Imperative Programming

### Core Idea

Tell the computer **HOW** to do something, step by step. Focus on **changing state** through explicit commands.

### Characteristics

- Explicit sequence of statements
- Variables that change over time (mutable state)
- Control flow: loops, conditionals
- Think: "Do this, then do that, then do this other thing"

### Example: Summing Array Elements

**Python:**
```python
def sum_array(numbers):
    total = 0  # Initial state
    for num in numbers:
        total = total + num  # Mutate state
    return total

result = sum_array([1, 2, 3, 4, 5])
print(result)  # Output: 15
```

**JavaScript:**
```javascript
function sumArray(numbers) {
    let total = 0;  // Initial state
    for (let i = 0; i < numbers.length; i++) {
        total = total + numbers[i];  // Mutate state
    }
    return total;
}

console.log(sumArray([1, 2, 3, 4, 5]));  // Output: 15
```

**Java:**
```java
public static int sumArray(int[] numbers) {
    int total = 0;  // Initial state
    for (int num : numbers) {
        total = total + num;  // Mutate state
    }
    return total;
}
```

**C++:**
```cpp
int sumArray(const std::vector<int>& numbers) {
    int total = 0;  // Initial state
    for (int num : numbers) {
        total = total + num;  // Mutate state
    }
    return total;
}
```

**Focus**: We explicitly manage the `total` variable, updating it with each iteration.

---

## Procedural Programming

### Core Idea

Organize imperative code into **procedures** (also called functions or subroutines). Structure programs as a collection of procedures that call each other.

### Characteristics

- Top-down design: break problems into procedures
- Each procedure performs a specific task
- Procedures can call other procedures
- Data and procedures are separate
- Emphasizes **modularity** and **reusability**

### Example: Structured Program with Functions

**Python:**
```python
def get_numbers():
    """Get numbers from user input"""
    return [1, 2, 3, 4, 5]

def calculate_sum(numbers):
    """Calculate sum of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total

def calculate_average(numbers):
    """Calculate average of numbers"""
    return calculate_sum(numbers) / len(numbers)

def display_results(numbers):
    """Display results"""
    total = calculate_sum(numbers)
    avg = calculate_average(numbers)
    print(f"Sum: {total}")
    print(f"Average: {avg}")

def main():
    """Main procedure that orchestrates the program"""
    numbers = get_numbers()
    display_results(numbers)

# Entry point
main()
```

**C:**
```c
#include <stdio.h>

// Procedures as separate functions
int calculate_sum(int numbers[], int size) {
    int total = 0;
    for (int i = 0; i < size; i++) {
        total += numbers[i];
    }
    return total;
}

double calculate_average(int numbers[], int size) {
    return (double)calculate_sum(numbers, size) / size;
}

void display_results(int numbers[], int size) {
    int total = calculate_sum(numbers, size);
    double avg = calculate_average(numbers, size);
    printf("Sum: %d\n", total);
    printf("Average: %.2f\n", avg);
}

int main() {
    int numbers[] = {1, 2, 3, 4, 5};
    int size = 5;
    display_results(numbers, size);
    return 0;
}
```

**Benefits**: Code is organized, reusable, and easier to test. Each procedure has a single responsibility.

---

## Object-Oriented Programming (OOP)

### Core Idea

Organize code around **objects** — bundles of data (attributes) and behavior (methods). Model real-world entities as objects that interact.

### Key Concepts

1. **Encapsulation**: Bundling data and methods together; hiding internal details
2. **Inheritance**: Creating new classes based on existing ones
3. **Polymorphism**: Same interface, different implementations
4. **Abstraction**: Defining interfaces without specifying implementation

### Example: Modeling a Bank Account

**Python:**
```python
class BankAccount:
    """Represents a bank account"""

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance  # Private attribute (encapsulation)

    def deposit(self, amount):
        """Deposit money"""
        if amount > 0:
            self.__balance += amount
            return True
        return False

    def withdraw(self, amount):
        """Withdraw money"""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False

    def get_balance(self):
        """Get current balance"""
        return self.__balance


class SavingsAccount(BankAccount):
    """Savings account with interest (inheritance)"""

    def __init__(self, owner, balance=0, interest_rate=0.02):
        super().__init__(owner, balance)
        self.interest_rate = interest_rate

    def apply_interest(self):
        """Apply interest to balance"""
        interest = self.get_balance() * self.interest_rate
        self.deposit(interest)


# Usage
account = SavingsAccount("Alice", 1000)
account.deposit(500)
account.apply_interest()
print(f"Balance: ${account.get_balance():.2f}")  # Output: Balance: $1530.00
```

**Java:**
```java
public class BankAccount {
    private String owner;
    private double balance;  // Encapsulation: private field

    public BankAccount(String owner, double balance) {
        this.owner = owner;
        this.balance = balance;
    }

    public boolean deposit(double amount) {
        if (amount > 0) {
            this.balance += amount;
            return true;
        }
        return false;
    }

    public boolean withdraw(double amount) {
        if (amount > 0 && amount <= this.balance) {
            this.balance -= amount;
            return true;
        }
        return false;
    }

    public double getBalance() {
        return this.balance;
    }
}

// Inheritance
public class SavingsAccount extends BankAccount {
    private double interestRate;

    public SavingsAccount(String owner, double balance, double interestRate) {
        super(owner, balance);
        this.interestRate = interestRate;
    }

    public void applyInterest() {
        double interest = getBalance() * interestRate;
        deposit(interest);
    }
}
```

**C++:**
```cpp
class BankAccount {
private:
    std::string owner;
    double balance;  // Encapsulation

public:
    BankAccount(const std::string& owner, double balance = 0)
        : owner(owner), balance(balance) {}

    bool deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            return true;
        }
        return false;
    }

    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }

    double getBalance() const {
        return balance;
    }
};

// Inheritance
class SavingsAccount : public BankAccount {
private:
    double interestRate;

public:
    SavingsAccount(const std::string& owner, double balance = 0, double interestRate = 0.02)
        : BankAccount(owner, balance), interestRate(interestRate) {}

    void applyInterest() {
        double interest = getBalance() * interestRate;
        deposit(interest);
    }
};
```

**Benefits**: Models real-world entities, promotes code reuse through inheritance, encapsulates complexity.

---

## Functional Programming (FP)

### Core Idea

Treat computation as the evaluation of mathematical functions. Avoid changing state and mutable data.

### Key Principles

1. **Pure Functions**: Same input always produces same output; no side effects
2. **Immutability**: Data cannot be changed after creation
3. **First-Class Functions**: Functions are values (can be passed, returned)
4. **Higher-Order Functions**: Functions that take/return other functions
5. **Declarative**: Express **WHAT** to compute, not **HOW**

### Example: Imperative vs Functional

**Imperative approach (Python):**
```python
# Mutable state, explicit loops
def get_even_squares(numbers):
    result = []
    for num in numbers:
        if num % 2 == 0:
            result.append(num ** 2)
    return result

print(get_even_squares([1, 2, 3, 4, 5, 6]))  # [4, 16, 36]
```

**Functional approach (Python):**
```python
# Immutable, declarative, higher-order functions
def get_even_squares(numbers):
    return list(
        map(lambda x: x ** 2,
            filter(lambda x: x % 2 == 0, numbers))
    )

# Or with list comprehension (more Pythonic)
def get_even_squares(numbers):
    return [x ** 2 for x in numbers if x % 2 == 0]

print(get_even_squares([1, 2, 3, 4, 5, 6]))  # [4, 16, 36]
```

**JavaScript (functional style):**
```javascript
const getEvenSquares = (numbers) =>
    numbers
        .filter(x => x % 2 === 0)  // Keep even numbers
        .map(x => x ** 2);         // Square them

console.log(getEvenSquares([1, 2, 3, 4, 5, 6]));  // [4, 16, 36]
```

### Pure Functions

**Impure function (has side effects):**
```python
total = 0  # External state

def add_to_total(x):
    global total
    total += x  # Modifies external state (side effect)
    return total
```

**Pure function (no side effects):**
```python
def add(x, y):
    return x + y  # Only depends on inputs, no external state
```

### Benefits of FP

- **Predictable**: Pure functions always return the same output for the same input
- **Testable**: Easy to test (no hidden state)
- **Parallelizable**: No shared state means safe concurrency
- **Composable**: Combine small functions to build complex behavior

---

## Declarative Programming

### Core Idea

Describe **WHAT** you want, not **HOW** to achieve it. The system figures out the implementation.

### Examples

**SQL (declarative query):**
```sql
-- You describe WHAT you want
SELECT name, age
FROM users
WHERE age > 18
ORDER BY name;

-- You don't specify HOW to:
-- - Scan the table
-- - Apply the filter
-- - Sort the results
-- The database engine decides the execution plan
```

**HTML (declarative structure):**
```html
<!-- You describe WHAT the page should look like -->
<html>
  <body>
    <h1>Welcome</h1>
    <p>This is a paragraph.</p>
  </body>
</html>

<!-- You don't specify HOW to:
     - Render pixels
     - Apply default styles
     - Handle layout
     The browser does that
-->
```

**CSS (declarative styling):**
```css
/* Describe WHAT styles you want */
.button {
    background-color: blue;
    color: white;
    padding: 10px;
}

/* Browser handles HOW to apply styles */
```

**Contrast with Imperative (JavaScript):**
```javascript
// Imperative: HOW to create and style a button
const button = document.createElement('button');
button.style.backgroundColor = 'blue';
button.style.color = 'white';
button.style.padding = '10px';
button.textContent = 'Click me';
document.body.appendChild(button);
```

---

## Logic Programming

### Core Idea

Express logic as facts and rules. The system performs logical inference to answer queries.

### Example: Prolog

```prolog
% Facts
parent(tom, bob).
parent(tom, liz).
parent(bob, ann).
parent(bob, pat).
parent(pat, jim).

% Rules
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% Queries
?- grandparent(tom, ann).  % true
?- ancestor(tom, jim).     % true
```

You define **WHAT is true**, and Prolog figures out **HOW to find answers** using backtracking and unification.

**Use cases**: Expert systems, natural language processing, theorem proving.

---

## Event-Driven Programming

### Core Idea

Program flow is determined by **events** (user actions, messages, sensor inputs). The system responds to events via callbacks or event handlers.

### Characteristics

- **Event loop**: Continuously checks for events
- **Callbacks/Handlers**: Functions executed in response to events
- **Asynchronous**: Events can occur at any time

### Example: GUI Programming

**JavaScript (browser):**
```javascript
// Event handler
function handleClick(event) {
    console.log('Button clicked!');
    console.log('Mouse position:', event.clientX, event.clientY);
}

// Register event listener
const button = document.getElementById('myButton');
button.addEventListener('click', handleClick);

// The event loop waits for events and calls handlers
```

**Python (with tkinter):**
```python
import tkinter as tk

def on_button_click():
    print("Button clicked!")

# Create window
root = tk.Tk()

# Create button with event handler
button = tk.Button(root, text="Click Me", command=on_button_click)
button.pack()

# Start event loop
root.mainloop()  # Waits for events (clicks, key presses, etc.)
```

**Use cases**: GUIs, web applications, game engines, IoT systems.

---

## Reactive Programming

### Core Idea

Programming with **asynchronous data streams**. React to changes over time.

### Characteristics

- **Observables**: Streams of data/events over time
- **Observers**: Subscribe to observables and react to emissions
- **Operators**: Transform, filter, combine streams

### Example: RxJS (JavaScript)

```javascript
import { fromEvent, interval } from 'rxjs';
import { map, filter, debounceTime } from 'rxjs/operators';

// Stream of click events
const clicks$ = fromEvent(document, 'click');

// Transform stream: get click positions
clicks$
    .pipe(
        map(event => ({ x: event.clientX, y: event.clientY }))
    )
    .subscribe(pos => console.log('Clicked at:', pos));

// Stream of time ticks
const ticks$ = interval(1000);  // Every second

ticks$
    .pipe(
        filter(n => n % 2 === 0)  // Only even numbers
    )
    .subscribe(n => console.log('Even tick:', n));

// Search input with debounce
const searchInput = document.getElementById('search');
const search$ = fromEvent(searchInput, 'input');

search$
    .pipe(
        map(event => event.target.value),
        debounceTime(300)  // Wait for 300ms pause
    )
    .subscribe(query => console.log('Search:', query));
```

**Use cases**: Real-time data (stock tickers, chat), user input handling, complex async workflows.

---

## Multi-Paradigm Languages

Most modern languages support **multiple paradigms**, giving you flexibility to choose the best approach for each problem.

### Python: Imperative, OOP, Functional

```python
# Imperative
total = 0
for i in range(10):
    total += i

# Object-Oriented
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

# Functional
from functools import reduce
total = reduce(lambda acc, x: acc + x, range(10), 0)
```

### JavaScript: Imperative, OOP, Functional, Event-Driven

```javascript
// Imperative
let total = 0;
for (let i = 0; i < 10; i++) {
    total += i;
}

// Object-Oriented (class syntax)
class Counter {
    constructor() {
        this.count = 0;
    }
    increment() {
        this.count++;
    }
}

// Functional
const total = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    .reduce((acc, x) => acc + x, 0);

// Event-Driven
button.addEventListener('click', () => console.log('Clicked'));
```

### Scala: OOP, Functional

```scala
// Object-Oriented
class BankAccount(var balance: Double) {
    def deposit(amount: Double): Unit = {
        balance += amount
    }
}

// Functional
val numbers = List(1, 2, 3, 4, 5)
val doubled = numbers.map(_ * 2)
val sum = numbers.reduce(_ + _)
```

---

## Comparing Paradigms: When to Use Which?

### Imperative/Procedural

**When:**
- Performance-critical code
- Low-level systems programming
- Algorithms that naturally involve state changes

**Example**: Device drivers, game engines, embedded systems

### Object-Oriented

**When:**
- Modeling complex domains with many entities
- Code reuse through inheritance
- Large teams and codebases (encapsulation helps manage complexity)

**Example**: Enterprise applications, GUI frameworks, simulations

### Functional

**When:**
- Data transformations
- Concurrent/parallel processing (no shared state)
- Mathematical computations
- Predictable, testable code

**Example**: Data pipelines, stream processing, compilers, financial systems

### Declarative

**When:**
- You want to express intent without implementation details
- The framework/library provides efficient implementation

**Example**: Database queries (SQL), UI markup (HTML/CSS), configuration (YAML)

### Event-Driven

**When:**
- User interfaces
- Asynchronous I/O
- Real-time systems

**Example**: Web apps, mobile apps, IoT, network servers

---

## Trade-Offs

| Paradigm       | Strengths                                      | Weaknesses                                   |
|----------------|------------------------------------------------|----------------------------------------------|
| Imperative     | Explicit control, performance                  | Verbose, harder to reason about state        |
| OOP            | Models real-world, encapsulation, reuse        | Can be over-engineered, inheritance issues   |
| Functional     | Predictable, testable, concurrency-safe        | Learning curve, performance overhead (immutability) |
| Declarative    | Concise, clear intent                          | Less control over execution                  |
| Event-Driven   | Responsive, asynchronous                       | Can be hard to debug, "callback hell"        |

---

## Exercises

### Exercise 1: Paradigm Recognition

Identify the paradigm(s) used in each code snippet:

**A)**
```python
numbers = [1, 2, 3, 4, 5]
result = sum(filter(lambda x: x % 2 == 0, numbers))
```

**B)**
```java
public class Car {
    private int speed;
    public void accelerate() {
        speed += 10;
    }
}
```

**C)**
```javascript
document.getElementById('btn').addEventListener('click', function() {
    alert('Clicked!');
});
```

### Exercise 2: Imperative to Functional

Rewrite this imperative code in a functional style:

```python
def process_data(numbers):
    result = []
    for num in numbers:
        if num > 0:
            result.append(num * 2)
    return result
```

### Exercise 3: OOP Design

Design classes for a simple library system:
- `Book`: title, author, ISBN
- `Member`: name, member ID, borrowed books
- `Library`: collection of books, members, borrow/return methods

Implement in your language of choice. Apply encapsulation, consider inheritance (maybe `EBook` extends `Book`?).

### Exercise 4: Functional vs OOP

Solve this problem using both OOP and functional approaches:

**Problem**: Calculate statistics (sum, average, max, min) for a list of numbers.

- **OOP**: Create a `Statistics` class
- **Functional**: Use pure functions and higher-order functions

Which approach feels more natural? Why?

### Exercise 5: Multi-Paradigm

Write a program that:
1. Defines a `Person` class (OOP)
2. Uses a list of `Person` objects
3. Filters adults (age >= 18) using functional programming
4. Sorts by name using a built-in function
5. Prints results imperatively with a loop

### Exercise 6: Event-Driven

Write a simple event-driven program (web or GUI):
- Create a text input and button
- On button click, display the input text in an alert/label
- (Bonus) Add a counter that increments with each click

---

## Summary

Programming paradigms are different **ways of thinking** about code:

- **Imperative**: Step-by-step commands, mutable state
- **Procedural**: Organizing imperative code into functions
- **Object-Oriented**: Modeling with objects, encapsulation, inheritance
- **Functional**: Pure functions, immutability, composition
- **Declarative**: Describe WHAT, not HOW
- **Logic**: Facts, rules, queries, inference
- **Event-Driven**: Responding to events asynchronously
- **Reactive**: Programming with data streams

**Key Insight**: No single paradigm is "best" — choose based on the problem, team, and ecosystem. Modern languages support multiple paradigms, giving you flexibility.

---

**Previous**: [What Is Programming](./01_What_Is_Programming.md) | **Next**: [Data Types & Abstraction](./03_Data_Types_and_Abstraction.md)
