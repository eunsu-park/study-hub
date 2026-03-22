# Control Flow Patterns

**Previous**: [Data Types & Abstraction](./03_Data_Types_and_Abstraction.md) | **Next**: [OOP Principles](./05_OOP_Principles.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain sequential execution as the default control flow and identify when it is insufficient
2. Apply conditional branching (if/else, switch/match, ternary, guard clauses) to direct program execution
3. Choose the appropriate loop construct (for, while, do-while) for a given repetition task and use break/continue effectively
4. State and verify loop invariants to reason about loop correctness
5. Implement recursive solutions with proper base and recursive cases, and convert between recursive and iterative forms
6. Use iterators and generators for lazy, memory-efficient data processing
7. Compare exception-based error handling (try/catch/finally) with Result/Either types for explicit error flow
8. Explain short-circuit evaluation and structured programming principles

---

Control flow determines the order in which a program's statements execute. Beyond the default top-to-bottom sequence, every nontrivial program needs decisions, repetition, and mechanisms for handling exceptional situations. Mastering these patterns is essential because they are the connective tissue of every algorithm -- the way you express *when*, *how many times*, and *under what conditions* each piece of logic runs.

## Sequential Execution

By default, programs execute **sequentially** — one statement after another, from top to bottom.

**Python:**
```python
print("Step 1")
x = 10
print("Step 2")
y = x + 5
print("Step 3")
print(f"Result: {y}")

# Output:
# Step 1
# Step 2
# Step 3
# Result: 15
```

**JavaScript:**
```javascript
console.log("Step 1");
let x = 10;
console.log("Step 2");
let y = x + 5;
console.log("Step 3");
console.log("Result:", y);
```

**Java:**
```java
System.out.println("Step 1");
int x = 10;
System.out.println("Step 2");
int y = x + 5;
System.out.println("Step 3");
System.out.println("Result: " + y);
```

This is the simplest form of control flow. But real programs need **decisions** and **repetition**.

---

## Conditional Branching

Execute different code paths based on conditions.

### If/Else

**Python:**
```python
age = 18

if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")
```

**JavaScript:**
```javascript
let age = 18;

if (age >= 18) {
    console.log("Adult");
} else if (age >= 13) {
    console.log("Teenager");
} else {
    console.log("Child");
}
```

**Java:**
```java
int age = 18;

if (age >= 18) {
    System.out.println("Adult");
} else if (age >= 13) {
    System.out.println("Teenager");
} else {
    System.out.println("Child");
}
```

**C++:**
```cpp
int age = 18;

if (age >= 18) {
    std::cout << "Adult" << std::endl;
} else if (age >= 13) {
    std::cout << "Teenager" << std::endl;
} else {
    std::cout << "Child" << std::endl;
}
```

### Switch/Match

For multiple discrete cases.

**Java (switch):**
```java
int day = 3;
String dayName;

switch (day) {
    case 1:
        dayName = "Monday";
        break;
    case 2:
        dayName = "Tuesday";
        break;
    case 3:
        dayName = "Wednesday";
        break;
    default:
        dayName = "Unknown";
        break;
}

System.out.println(dayName);  // Wednesday
```

**JavaScript (switch):**
```javascript
let day = 3;
let dayName;

switch (day) {
    case 1:
        dayName = "Monday";
        break;
    case 2:
        dayName = "Tuesday";
        break;
    case 3:
        dayName = "Wednesday";
        break;
    default:
        dayName = "Unknown";
}

console.log(dayName);  // Wednesday
```

**C++ (switch):**
```cpp
int day = 3;
std::string dayName;

switch (day) {
    case 1:
        dayName = "Monday";
        break;
    case 2:
        dayName = "Tuesday";
        break;
    case 3:
        dayName = "Wednesday";
        break;
    default:
        dayName = "Unknown";
        break;
}

std::cout << dayName << std::endl;  // Wednesday
```

### Pattern Matching (Modern Languages)

**Python (3.10+):**
```python
def describe(value):
    match value:
        case 0:
            return "zero"
        case 1 | 2 | 3:
            return "small"
        case int(x) if x < 0:
            return "negative"
        case int():
            return "positive integer"
        case str():
            return "string"
        case _:
            return "unknown"

print(describe(2))     # small
print(describe(-5))    # negative
print(describe("hi"))  # string
```

**Rust (match):**
```rust
fn describe(value: i32) -> &'static str {
    match value {
        0 => "zero",
        1..=3 => "small",
        x if x < 0 => "negative",
        _ => "positive",
    }
}

println!("{}", describe(2));   // small
println!("{}", describe(-5));  // negative
```

**Benefits of pattern matching**: More expressive, exhaustiveness checking (compiler ensures all cases covered).

### Ternary Operator

Concise conditional expression.

**Python:**
```python
age = 20
status = "adult" if age >= 18 else "minor"
print(status)  # adult
```

**JavaScript:**
```javascript
let age = 20;
let status = age >= 18 ? "adult" : "minor";
console.log(status);  // adult
```

**Java:**
```java
int age = 20;
String status = age >= 18 ? "adult" : "minor";
System.out.println(status);  // adult
```

**C++:**
```cpp
int age = 20;
std::string status = age >= 18 ? "adult" : "minor";
std::cout << status << std::endl;  // adult
```

### Guard Clauses

**Early returns** to reduce nesting and improve readability.

**Before (nested):**
```python
def process_user(user):
    if user is not None:
        if user.is_active:
            if user.has_permission("admin"):
                print("Processing admin user")
                # ... complex logic ...
            else:
                print("No permission")
        else:
            print("Inactive user")
    else:
        print("No user")
```

**After (guard clauses):**
```python
def process_user(user):
    if user is None:
        print("No user")
        return

    if not user.is_active:
        print("Inactive user")
        return

    if not user.has_permission("admin"):
        print("No permission")
        return

    print("Processing admin user")
    # ... complex logic ...
```

**Benefits**: Reduced nesting, clearer error handling, main logic at end.

**JavaScript:**
```javascript
function processUser(user) {
    if (!user) {
        console.log("No user");
        return;
    }

    if (!user.isActive) {
        console.log("Inactive user");
        return;
    }

    if (!user.hasPermission("admin")) {
        console.log("No permission");
        return;
    }

    console.log("Processing admin user");
    // ... main logic ...
}
```

---

## Loops

Repetition is fundamental to programming.

### For Loop

Iterate a fixed number of times or over a collection.

**Python:**
```python
# Range-based
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# Iterating over collection
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# With index
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```

**JavaScript:**
```javascript
// Traditional for loop
for (let i = 0; i < 5; i++) {
    console.log(i);  // 0, 1, 2, 3, 4
}

// For-of (ES6)
let fruits = ["apple", "banana", "cherry"];
for (let fruit of fruits) {
    console.log(fruit);
}

// For-in (iterates over keys/indices)
for (let index in fruits) {
    console.log(index, fruits[index]);
}
```

**Java:**
```java
// Traditional for loop
for (int i = 0; i < 5; i++) {
    System.out.println(i);  // 0, 1, 2, 3, 4
}

// Enhanced for loop (for-each)
String[] fruits = {"apple", "banana", "cherry"};
for (String fruit : fruits) {
    System.out.println(fruit);
}
```

**C++:**
```cpp
// Traditional for loop
for (int i = 0; i < 5; i++) {
    std::cout << i << std::endl;  // 0, 1, 2, 3, 4
}

// Range-based for loop (C++11)
std::vector<std::string> fruits = {"apple", "banana", "cherry"};
for (const auto& fruit : fruits) {
    std::cout << fruit << std::endl;
}
```

### While Loop

Repeat while a condition is true.

**Python:**
```python
count = 0
while count < 5:
    print(count)
    count += 1
```

**JavaScript:**
```javascript
let count = 0;
while (count < 5) {
    console.log(count);
    count++;
}
```

**Java:**
```java
int count = 0;
while (count < 5) {
    System.out.println(count);
    count++;
}
```

### Do-While Loop

Execute at least once, then repeat while condition is true.

**Java:**
```java
int count = 0;
do {
    System.out.println(count);
    count++;
} while (count < 5);

// Executes at least once even if condition is false
int x = 10;
do {
    System.out.println("Runs once");
} while (x < 5);  // Still runs once
```

**JavaScript:**
```javascript
let count = 0;
do {
    console.log(count);
    count++;
} while (count < 5);
```

**C++:**
```cpp
int count = 0;
do {
    std::cout << count << std::endl;
    count++;
} while (count < 5);
```

**Note**: Python doesn't have a do-while loop. Use `while True` with `break`:
```python
count = 0
while True:
    print(count)
    count += 1
    if count >= 5:
        break
```

### Loop Control: Break and Continue

**Break**: Exit the loop immediately.

**Python:**
```python
for i in range(10):
    if i == 5:
        break  # Exit loop when i is 5
    print(i)  # 0, 1, 2, 3, 4
```

**Continue**: Skip the rest of the current iteration, proceed to next.

**Python:**
```python
for i in range(10):
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i)  # 1, 3, 5, 7, 9
```

**JavaScript:**
```javascript
for (let i = 0; i < 10; i++) {
    if (i === 5) break;  // Exit at 5
    console.log(i);  // 0, 1, 2, 3, 4
}

for (let i = 0; i < 10; i++) {
    if (i % 2 === 0) continue;  // Skip even
    console.log(i);  // 1, 3, 5, 7, 9
}
```

### Loop Invariants

A **loop invariant** is a condition that is true before and after each iteration. Useful for reasoning about correctness.

**Example: Finding maximum in array**

**Invariant**: At the start of each iteration, `max` holds the maximum of all elements examined so far.

**Python:**
```python
def find_max(numbers):
    if not numbers:
        return None

    max_val = numbers[0]  # Invariant: max_val is max of numbers[0:0+1]

    for i in range(1, len(numbers)):
        # Invariant: max_val is max of numbers[0:i]
        if numbers[i] > max_val:
            max_val = numbers[i]
        # Invariant maintained: max_val is max of numbers[0:i+1]

    return max_val
```

Understanding invariants helps you write correct loops and debug when things go wrong.

---

## Recursion

A function that calls itself. Every recursive function needs:
1. **Base case**: Condition to stop recursion
2. **Recursive case**: Call itself with a simpler input

### Factorial

**Mathematical definition**:
- `factorial(0) = 1` (base case)
- `factorial(n) = n × factorial(n-1)` (recursive case)

**Python:**
```python
def factorial(n):
    if n == 0:
        return 1  # Base case
    else:
        return n * factorial(n - 1)  # Recursive case

print(factorial(5))  # 120 (5 × 4 × 3 × 2 × 1)
```

**JavaScript:**
```javascript
function factorial(n) {
    if (n === 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

console.log(factorial(5));  // 120
```

**Java:**
```java
public static int factorial(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

System.out.println(factorial(5));  // 120
```

**C++:**
```cpp
int factorial(int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

std::cout << factorial(5) << std::endl;  // 120
```

### Fibonacci

**Definition**:
- `fib(0) = 0`, `fib(1) = 1` (base cases)
- `fib(n) = fib(n-1) + fib(n-2)` (recursive case)

**Python:**
```python
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(6))  # 8 (0, 1, 1, 2, 3, 5, 8)
```

**Note**: This is inefficient (exponential time) due to repeated calculations. Use memoization or iteration for better performance.

### Tree Traversal

Recursion shines with tree structures.

**Python:**
```python
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal(node):
    """Left → Root → Right"""
    if node is None:
        return  # Base case

    inorder_traversal(node.left)
    print(node.value)
    inorder_traversal(node.right)

# Example tree:
#       1
#      / \
#     2   3
#    / \
#   4   5

root = TreeNode(1,
    TreeNode(2, TreeNode(4), TreeNode(5)),
    TreeNode(3)
)

inorder_traversal(root)  # Output: 4, 2, 5, 1, 3
```

### Tail Recursion

A recursive call is **tail-recursive** if it's the last operation in the function. Some compilers optimize tail recursion into loops (no stack growth).

**Not tail-recursive (factorial):**
```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)  # Multiplication AFTER recursive call
```

**Tail-recursive (factorial with accumulator):**
```python
def factorial_tail(n, acc=1):
    if n == 0:
        return acc
    else:
        return factorial_tail(n - 1, n * acc)  # Recursive call is last operation

print(factorial_tail(5))  # 120
```

**Languages with tail-call optimization**: Scheme, Scala, some Rust, some JavaScript engines.

### When to Use Recursion vs Iteration

**Recursion is better when**:
- Problem is naturally recursive (trees, graphs, divide-and-conquer)
- Code is clearer, more elegant

**Iteration is better when**:
- Performance matters (avoid stack overhead)
- Problem is naturally iterative (simple loops)

**Example: Factorial iteratively**

**Python:**
```python
def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(factorial_iterative(5))  # 120
```

More efficient (no recursive calls), but recursion is often clearer for complex problems.

---

## Iterators and Generators

### Iterators

Objects that produce a sequence of values one at a time.

**Python:**
```python
# Lists are iterable
numbers = [1, 2, 3, 4, 5]
iterator = iter(numbers)

print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
```

**JavaScript:**
```javascript
let numbers = [1, 2, 3, 4, 5];
let iterator = numbers[Symbol.iterator]();

console.log(iterator.next().value);  // 1
console.log(iterator.next().value);  // 2
console.log(iterator.next().value);  // 3
```

**Benefits**: Memory-efficient (don't need entire collection in memory), lazy evaluation.

### Generators

Functions that **yield** values one at a time, pausing between yields.

**Python:**
```python
def count_up_to(n):
    count = 1
    while count <= n:
        yield count  # Pause here, return count
        count += 1

# Create generator
gen = count_up_to(5)

print(next(gen))  # 1
print(next(gen))  # 2

# Or use in loop
for num in count_up_to(5):
    print(num)  # 1, 2, 3, 4, 5
```

**JavaScript:**
```javascript
function* countUpTo(n) {
    let count = 1;
    while (count <= n) {
        yield count;
        count++;
    }
}

let gen = countUpTo(5);
console.log(gen.next().value);  // 1
console.log(gen.next().value);  // 2

// Or use in loop
for (let num of countUpTo(5)) {
    console.log(num);  // 1, 2, 3, 4, 5
}
```

**Benefits**:
- **Lazy evaluation**: Values computed on-demand
- **Memory-efficient**: Don't store entire sequence
- **Infinite sequences**: Can represent infinite streams

**Example: Infinite sequence**

**Python:**
```python
def infinite_count():
    count = 0
    while True:
        yield count
        count += 1

# Only compute as needed
gen = infinite_count()
print(next(gen))  # 0
print(next(gen))  # 1
print(next(gen))  # 2
# ... can continue forever
```

---

## Coroutines and Async/Await

**Coroutines**: Functions that can pause and resume, enabling cooperative multitasking.

### Async/Await Pattern

**Python:**
```python
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}...")
    await asyncio.sleep(2)  # Simulate network delay
    print(f"Done fetching {url}")
    return f"Data from {url}"

async def main():
    # Run concurrently
    task1 = fetch_data("https://api1.com")
    task2 = fetch_data("https://api2.com")

    result1, result2 = await asyncio.gather(task1, task2)
    print(result1, result2)

# Run
asyncio.run(main())
```

**JavaScript:**
```javascript
async function fetchData(url) {
    console.log(`Fetching ${url}...`);
    await new Promise(resolve => setTimeout(resolve, 2000));  // Simulate delay
    console.log(`Done fetching ${url}`);
    return `Data from ${url}`;
}

async function main() {
    let task1 = fetchData("https://api1.com");
    let task2 = fetchData("https://api2.com");

    let [result1, result2] = await Promise.all([task1, task2]);
    console.log(result1, result2);
}

main();
```

**Key difference from threads**: Coroutines are **cooperative** (they explicitly yield control), not **preemptive** (OS can interrupt threads at any time).

---

## Error Flow

How do you handle errors and exceptional situations?

### Exceptions

**Try/Catch/Finally**

**Python:**
```python
def divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return None
    finally:
        print("Cleanup (always runs)")

divide(10, 2)   # 5.0, "Cleanup"
divide(10, 0)   # "Error", None, "Cleanup"
```

**JavaScript:**
```javascript
function divide(a, b) {
    try {
        if (b === 0) {
            throw new Error("Cannot divide by zero");
        }
        return a / b;
    } catch (error) {
        console.log("Error:", error.message);
        return null;
    } finally {
        console.log("Cleanup (always runs)");
    }
}

divide(10, 2);  // 5, "Cleanup"
divide(10, 0);  // "Error: Cannot divide by zero", null, "Cleanup"
```

**Java:**
```java
public static Double divide(int a, int b) {
    try {
        return (double) a / b;
    } catch (ArithmeticException e) {
        System.out.println("Error: " + e.getMessage());
        return null;
    } finally {
        System.out.println("Cleanup (always runs)");
    }
}
```

**C++:**
```cpp
double divide(int a, int b) {
    try {
        if (b == 0) {
            throw std::runtime_error("Cannot divide by zero");
        }
        return static_cast<double>(a) / b;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 0.0;
    }
}
```

### Result/Either Types

Functional approach: return a type that represents success or failure.

**Rust:**
```rust
fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        Err("Cannot divide by zero".to_string())
    } else {
        Ok(a / b)
    }
}

match divide(10, 2) {
    Ok(result) => println!("Result: {}", result),
    Err(error) => println!("Error: {}", error),
}
```

**Benefits**: Errors are explicit in the type signature. Compiler forces you to handle them.

### Error Propagation

**Rust's `?` operator**:
```rust
fn read_file_length(path: &str) -> Result<usize, std::io::Error> {
    let contents = std::fs::read_to_string(path)?;  // Propagate error if it occurs
    Ok(contents.len())
}
```

If `read_to_string` returns an `Err`, it's immediately returned from `read_file_length`. Otherwise, continue.

**Java's `throws`**:
```java
public static String readFile(String path) throws IOException {
    return new String(Files.readAllBytes(Paths.get(path)));
}

// Caller must handle
try {
    String content = readFile("file.txt");
} catch (IOException e) {
    System.out.println("Error reading file");
}
```

---

## Short-Circuit Evaluation

Logical operators `&&` (AND) and `||` (OR) use **short-circuit evaluation**: they stop evaluating as soon as the result is determined.

**Python:**
```python
def is_positive(x):
    print(f"Checking {x}")
    return x > 0

# AND: stops at first false
result = is_positive(5) and is_positive(10) and is_positive(-3)
# Output: Checking 5, Checking 10, Checking -3
# Result: False

# OR: stops at first true
result = is_positive(-5) or is_positive(10) or is_positive(20)
# Output: Checking -5, Checking 10
# Result: True (doesn't check 20)
```

**JavaScript:**
```javascript
function check(x) {
    console.log(`Checking ${x}`);
    return x > 0;
}

let result = check(5) && check(10) && check(-3);
// Logs: Checking 5, Checking 10, Checking -3

let result2 = check(-5) || check(10) || check(20);
// Logs: Checking -5, Checking 10
// Doesn't log Checking 20 (short-circuited)
```

**Use case: Avoiding null/undefined errors**

**JavaScript:**
```javascript
let user = getUser();
if (user && user.isActive && user.hasPermission("admin")) {
    console.log("Admin user");
}
// If user is null, doesn't try to access user.isActive (would error)
```

---

## Structured Programming

**Structured programming** (1960s-70s) advocated for:
- **No `goto`**: Use loops and functions instead
- **Single entry, single exit**: Functions have one entry point and one return (though multiple returns are common now)
- **Top-down design**: Break problems into smaller procedures

**Bad (unstructured with goto):**
```c
// Don't do this
int i = 0;
start:
    printf("%d\n", i);
    i++;
    if (i < 5) goto start;
```

**Good (structured with loop):**
```c
for (int i = 0; i < 5; i++) {
    printf("%d\n", i);
}
```

**Modern consensus**: Structured programming principles are good, but pragmatic use of early returns and `break`/`continue` improves readability.

---

## Exercises

### Exercise 1: Refactor with Guard Clauses

Refactor this nested code using guard clauses:

**Python:**
```python
def process_order(order):
    if order is not None:
        if order.is_valid():
            if order.total > 0:
                if order.user.is_verified:
                    print("Processing order")
                else:
                    print("User not verified")
            else:
                print("Order total must be positive")
        else:
            print("Invalid order")
    else:
        print("No order")
```

### Exercise 2: Recursion vs Iteration

Implement **sum of an array** both recursively and iteratively:

**Recursive:**
```python
def sum_recursive(numbers):
    # Base case: empty array
    # Recursive case: first element + sum of rest
    pass
```

**Iterative:**
```python
def sum_iterative(numbers):
    # Use a loop
    pass
```

Which is clearer? Which is more efficient?

### Exercise 3: Generator

Write a generator that produces the **Fibonacci sequence** indefinitely:

```python
def fibonacci_gen():
    # Yield 0, 1, 1, 2, 3, 5, 8, 13, ...
    pass

# Usage
gen = fibonacci_gen()
for i in range(10):
    print(next(gen))  # First 10 Fibonacci numbers
```

### Exercise 4: Pattern Matching

If your language supports pattern matching (Python 3.10+, Rust, Scala), write a function that classifies a value:
- If it's 0: "zero"
- If it's 1-10: "small"
- If it's 11-100: "medium"
- If it's > 100: "large"
- If it's negative: "negative"
- Otherwise: "unknown"

### Exercise 5: Error Handling

Write a function `safe_divide(a, b)` that:
- Returns the result of `a / b`
- Handles division by zero gracefully
- Uses exceptions (try/catch) in one implementation
- Uses a Result type (or similar) in another

**Python:**
```python
# Exception version
def safe_divide_exception(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

# Result version (using a tuple)
def safe_divide_result(a, b):
    if b == 0:
        return (False, "Cannot divide by zero")
    else:
        return (True, a / b)

# Usage
success, value = safe_divide_result(10, 2)
if success:
    print(f"Result: {value}")
else:
    print(f"Error: {value}")
```

### Exercise 6: Tail Recursion

Rewrite the **Fibonacci function** to be tail-recursive:

**Hint**: Use helper function with accumulators.

```python
def fibonacci_tail(n, a=0, b=1):
    # Base case: n == 0
    # Recursive case: call with updated accumulators
    pass
```

---

## Summary

Control flow determines the **order of execution**:

- **Sequential**: Default, top to bottom
- **Branching**:
  - If/else, switch/match, ternary operator
  - Guard clauses: early returns for clarity
- **Loops**:
  - For, while, do-while
  - Break/continue for control
  - Loop invariants for reasoning about correctness
- **Recursion**:
  - Base case + recursive case
  - Tail recursion for optimization
  - Use when problem is naturally recursive
- **Iterators & Generators**:
  - Lazy evaluation, memory-efficient
  - Produce values on-demand
- **Coroutines & Async/Await**:
  - Cooperative multitasking
  - Pause/resume execution
- **Error Handling**:
  - Exceptions: try/catch/finally
  - Result types: explicit error values
  - Error propagation: `?` operator, `throws`
- **Short-circuit evaluation**: `&&` and `||` optimize logic
- **Structured programming**: Avoid goto, use structured constructs

**Key Insight**: Choose the right control flow pattern for the problem. Guard clauses reduce nesting. Recursion shines for trees and graphs. Generators enable infinite sequences. Exceptions handle exceptional cases. Each pattern has its place.

---

**Previous**: [Data Types & Abstraction](./03_Data_Types_and_Abstraction.md) | **Next**: [OOP Principles](./05_OOP_Principles.md)
