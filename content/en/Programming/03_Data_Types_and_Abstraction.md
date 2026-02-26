# Data Types & Abstraction

**Previous**: [Programming Paradigms](./02_Programming_Paradigms.md) | **Next**: [Control Flow Patterns](./04_Control_Flow_Patterns.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a type is and why types are essential for preventing errors
2. Identify and use primitive types (integers, floats, booleans, characters, strings) across multiple languages
3. Construct composite types (arrays, structs/records, tuples) from primitive building blocks
4. Compare static vs. dynamic typing and strong vs. weak typing with concrete examples
5. Define Abstract Data Types (ADTs) and explain the distinction between interface and implementation
6. Implement classic ADTs (Stack, Queue, Map) using different underlying data structures
7. Apply generics and templates to write type-safe, reusable code
8. Describe algebraic data types (sum types, product types) and explain how Option/Maybe types address the null problem

---

Every value in a program has a type, and the way a language handles types profoundly affects how you write, debug, and reason about code. From the hardware-level distinction between an integer and a floating-point number to the design-level concept of an Abstract Data Type, types are the primary mechanism for preventing errors and managing complexity. This lesson builds your fluency with type systems and shows how abstraction lets you think at higher levels without losing rigor.

## What Are Types?

A **type** is a classification of data that determines:
- What values the data can hold
- What operations can be performed on it
- How much memory it occupies
- How it's interpreted by the computer

> **📌 Analogy:** Types are like containers -- a glass bottle holds liquids, a cardboard box holds solid items. You cannot pour water into a cardboard box (or rather, you should not). Types enforce similar constraints in code.

### Why Types Matter

```python
# Without types (conceptually):
x = "42"
y = 10
z = x + y  # What should this mean? "4210" or 52? Error?

# With types:
x: str = "42"
y: int = 10
# z = x + y  # Type error: can't add string and int
z = int(x) + y  # Explicit conversion: 52
```

Types prevent errors by enforcing **constraints** on what operations are valid.

---

## Primitive Types

Primitive types are the **building blocks** provided by the language. They typically map directly to hardware representations.

### Integers

Whole numbers without fractional components.

**Python:**
```python
age = 25
population = 7_800_000_000  # Python allows underscores for readability
negative = -42
```

**Java:**
```java
byte smallNumber = 127;       // 8-bit: -128 to 127
short mediumNumber = 32000;   // 16-bit: -32,768 to 32,767
int standardNumber = 100000;  // 32-bit: ~-2B to 2B
long largeNumber = 10000000000L;  // 64-bit
```

**C++:**
```cpp
int x = 42;
unsigned int y = 100;  // Only positive values
long long z = 9223372036854775807LL;  // 64-bit
```

### Floating-Point Numbers

Numbers with fractional parts. Approximate representations due to binary encoding.

**Python:**
```python
pi = 3.14159
scientific = 6.022e23  # 6.022 × 10^23 (Avogadro's number)
```

**JavaScript:**
```javascript
let price = 19.99;
let tiny = 0.0000001;
let notExact = 0.1 + 0.2;  // 0.30000000000000004 (floating-point precision issue)
```

**Java:**
```java
float f = 3.14f;     // 32-bit, single precision
double d = 3.14159;  // 64-bit, double precision (default)
```

### Booleans

True or false values for logical operations.

**Python:**
```python
is_active = True
has_permission = False

if is_active and has_permission:
    print("Access granted")
```

**JavaScript:**
```javascript
let isLoggedIn = true;
let isAdmin = false;
console.log(isLoggedIn && !isAdmin);  // true
```

**C++:**
```cpp
bool flag = true;
bool result = (5 > 3);  // true
```

### Characters

Single characters, often represented as integers (ASCII/Unicode code points).

**Java:**
```java
char letter = 'A';  // Single quotes for char
char unicode = '\u0041';  // Unicode: also 'A'
```

**C++:**
```cpp
char c = 'x';
char newline = '\n';
```

**Python:**
```python
# Python has no separate char type; single-character strings
letter = 'A'
```

### Strings

Sequences of characters. Some languages treat strings as primitives, others as objects.

**Python:**
```python
name = "Alice"
message = 'Hello, World!'
multiline = """This is
a multi-line
string"""
```

**JavaScript:**
```javascript
let greeting = "Hello";
let template = `Hello, ${name}!`;  // Template literals
```

**Java:**
```java
String text = "Hello, World!";  // String is an object, not primitive
```

**C++:**
```cpp
#include <string>
std::string message = "Hello, C++!";
```

---

## Composite Types

Types built from primitive types.

### Arrays

Fixed-size, ordered collections of elements of the same type.

**Python:**
```python
# Python lists are dynamic, not fixed-size, but conceptually similar
numbers = [1, 2, 3, 4, 5]
```

**JavaScript:**
```javascript
let numbers = [1, 2, 3, 4, 5];  // Dynamic arrays
```

**Java:**
```java
int[] numbers = {1, 2, 3, 4, 5};  // Fixed size
int[] array = new int[10];  // Allocate size 10, initialized to 0
```

**C++:**
```cpp
#include <array>
std::array<int, 5> numbers = {1, 2, 3, 4, 5};  // Fixed size 5
```

### Records/Structs

Group related data of different types.

**C:**
```c
struct Person {
    char name[50];
    int age;
    double salary;
};

struct Person alice = {"Alice", 30, 75000.0};
printf("%s is %d years old\n", alice.name, alice.age);
```

**C++:**
```cpp
struct Point {
    int x;
    int y;
};

Point p = {10, 20};
std::cout << "x: " << p.x << ", y: " << p.y << std::endl;
```

### Tuples

Ordered, fixed-size collections of elements, possibly of different types.

**Python:**
```python
person = ("Alice", 30, "Engineer")  # (name, age, job)
name, age, job = person  # Unpacking
```

**JavaScript (using arrays):**
```javascript
let person = ["Alice", 30, "Engineer"];
let [name, age, job] = person;  // Destructuring
```

---

## Type Systems

### Static vs Dynamic Typing

**Static Typing**: Types are checked at **compile time**. Variables have fixed types.

**Languages**: Java, C++, C, Rust, Go, TypeScript

**Java example:**
```java
int x = 10;
// x = "hello";  // Compile error: incompatible types
x = 20;  // OK
```

**C++ example:**
```cpp
int count = 5;
// count = "text";  // Compile error
count = 10;  // OK
```

**Benefits**:
- Catch errors early (before running the program)
- Better tooling (autocomplete, refactoring)
- Performance optimizations (compiler knows types)

**Trade-offs**:
- More verbose (type annotations)
- Less flexibility

---

**Dynamic Typing**: Types are checked at **runtime**. Variables can hold any type.

**Languages**: Python, JavaScript, Ruby, PHP

**Python example:**
```python
x = 10       # x is an int
x = "hello"  # Now x is a string — no error
x = [1, 2]   # Now x is a list
```

**JavaScript example:**
```javascript
let x = 10;
x = "hello";  // OK
x = {key: "value"};  // OK
```

**Benefits**:
- Less boilerplate
- More flexible
- Faster prototyping

**Trade-offs**:
- Errors caught at runtime (might crash in production)
- Harder to reason about code in large projects
- Slower performance (runtime type checks)

---

### Strong vs Weak Typing

**Strong Typing**: Strict enforcement of type rules. No implicit conversions between incompatible types.

**Python (strong):**
```python
x = "5"
y = 10
# z = x + y  # TypeError: can't add string and int
z = int(x) + y  # Must explicitly convert: 15
```

**Weak Typing**: Allows implicit type conversions (type coercion).

**JavaScript (weak):**
```javascript
let x = "5";
let y = 10;
let z = x + y;  // "510" — string concatenation (implicit conversion)
let w = x - y;  // -5 — subtraction (implicit conversion to number)
console.log(z, w);  // "510", -5
```

**Benefits of strong typing**: Fewer surprises, clearer intent
**Benefits of weak typing**: More permissive, less verbose (but more bugs)

---

### Type Inference

The compiler/interpreter deduces types automatically.

**Kotlin:**
```kotlin
val x = 10  // Inferred as Int
val name = "Alice"  // Inferred as String
// x = "text"  // Error: type mismatch
```

**Rust:**
```rust
let x = 10;  // Inferred as i32 (32-bit integer)
let y = 3.14;  // Inferred as f64 (64-bit float)
```

**TypeScript:**
```javascript
let count = 5;  // Inferred as number
// count = "text";  // Error: Type 'string' is not assignable to type 'number'
```

**Benefits**: Conciseness of dynamic typing + safety of static typing.

---

## Abstract Data Types (ADTs)

An **Abstract Data Type** separates the **interface** (what operations are available) from the **implementation** (how those operations are performed).

**Key idea**: Users interact with the ADT through a well-defined interface without needing to know internal details.

### Stack ADT

**Interface (operations)**:
- `push(item)`: Add item to top
- `pop()`: Remove and return top item
- `peek()`: View top item without removing
- `is_empty()`: Check if stack is empty

**Implementation 1: Using an array**

**Python:**
```python
class ArrayStack:
    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._data.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0


# Usage (same interface regardless of implementation)
stack = ArrayStack()
stack.push(10)
stack.push(20)
print(stack.pop())  # 20
```

**Implementation 2: Using a linked list**

**Python:**
```python
class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


class LinkedStack:
    def __init__(self):
        self._top = None
        self._size = 0

    def push(self, item):
        self._top = Node(item, self._top)
        self._size += 1

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        value = self._top.value
        self._top = self._top.next
        self._size -= 1
        return value

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._top.value

    def is_empty(self):
        return self._top is None


# Usage (same interface!)
stack = LinkedStack()
stack.push(10)
stack.push(20)
print(stack.pop())  # 20
```

**Key point**: The **interface** is the same. Users don't need to know if it's array-based or linked-list-based. This is **abstraction**.

---

### Queue ADT

**Interface**:
- `enqueue(item)`: Add to rear
- `dequeue()`: Remove and return from front
- `is_empty()`: Check if empty

**Java implementation:**
```java
import java.util.LinkedList;

public interface Queue<T> {
    void enqueue(T item);
    T dequeue();
    boolean isEmpty();
}

public class LinkedQueue<T> implements Queue<T> {
    private LinkedList<T> data = new LinkedList<>();

    public void enqueue(T item) {
        data.addLast(item);
    }

    public T dequeue() {
        if (isEmpty()) {
            throw new RuntimeException("Queue is empty");
        }
        return data.removeFirst();
    }

    public boolean isEmpty() {
        return data.isEmpty();
    }
}
```

---

### Map/Dictionary ADT

**Interface**:
- `put(key, value)`: Store key-value pair
- `get(key)`: Retrieve value by key
- `remove(key)`: Delete key-value pair
- `contains(key)`: Check if key exists

**Python (using built-in dict):**
```python
# Python's dict is an implementation of the Map ADT
phonebook = {}
phonebook["Alice"] = "555-1234"
phonebook["Bob"] = "555-5678"

print(phonebook["Alice"])  # "555-1234"
print("Alice" in phonebook)  # True
```

**JavaScript (using Map):**
```javascript
let map = new Map();
map.set("Alice", "555-1234");
map.set("Bob", "555-5678");

console.log(map.get("Alice"));  // "555-1234"
console.log(map.has("Alice"));  // true
```

---

## Generics and Templates

**Generics** (Java, C#, TypeScript) and **Templates** (C++) allow you to write code that works with **any type**.

### Java Generics

**Without generics:**
```java
// Must use Object, lose type safety
public class Box {
    private Object item;

    public void set(Object item) {
        this.item = item;
    }

    public Object get() {
        return item;
    }
}

Box box = new Box();
box.set("Hello");
String s = (String) box.get();  // Explicit cast needed
```

**With generics:**
```java
public class Box<T> {
    private T item;

    public void set(T item) {
        this.item = item;
    }

    public T get() {
        return item;
    }
}

Box<String> stringBox = new Box<>();
stringBox.set("Hello");
String s = stringBox.get();  // No cast needed, type-safe

Box<Integer> intBox = new Box<>();
intBox.set(42);
// intBox.set("text");  // Compile error
```

### C++ Templates

```cpp
template <typename T>
class Box {
private:
    T item;

public:
    void set(T value) {
        item = value;
    }

    T get() const {
        return item;
    }
};

// Usage
Box<int> intBox;
intBox.set(42);
std::cout << intBox.get() << std::endl;  // 42

Box<std::string> stringBox;
stringBox.set("Hello");
std::cout << stringBox.get() << std::endl;  // Hello
```

### TypeScript Generics

```javascript
class Box<T> {
    private item: T;

    set(value: T): void {
        this.item = value;
    }

    get(): T {
        return this.item;
    }
}

let stringBox = new Box<string>();
stringBox.set("Hello");
console.log(stringBox.get());  // Hello

let numberBox = new Box<number>();
numberBox.set(42);
console.log(numberBox.get());  // 42
```

**Benefits**: Code reuse, type safety, no runtime overhead (types erased in Java, template instantiation in C++).

---

## Algebraic Data Types

### Sum Types (Enums, Tagged Unions)

A value can be **one of several variants**.

**Rust enum:**
```rust
enum Status {
    Success,
    Error(String),
    Loading,
}

let result = Status::Error("Network timeout".to_string());

match result {
    Status::Success => println!("Success!"),
    Status::Error(msg) => println!("Error: {}", msg),
    Status::Loading => println!("Loading..."),
}
```

**TypeScript discriminated unions:**
```javascript
type Status =
    | { kind: "success"; data: string }
    | { kind: "error"; message: string }
    | { kind: "loading" };

function handleStatus(status: Status) {
    switch (status.kind) {
        case "success":
            console.log("Data:", status.data);
            break;
        case "error":
            console.log("Error:", status.message);
            break;
        case "loading":
            console.log("Loading...");
            break;
    }
}
```

### Product Types (Tuples, Records)

A value contains **multiple fields together**.

**Rust struct:**
```rust
struct Point {
    x: i32,
    y: i32,
}

let p = Point { x: 10, y: 20 };
println!("({}, {})", p.x, p.y);
```

**TypeScript:**
```javascript
type Point = {
    x: number;
    y: number;
};

let p: Point = { x: 10, y: 20 };
console.log(`(${p.x}, ${p.y})`);
```

---

## Null and Its Problems

**The Billion-Dollar Mistake** — Tony Hoare (inventor of null references):

> "I call it my billion-dollar mistake. It was the invention of the null reference in 1965... This has led to innumerable errors, vulnerabilities, and system crashes."

### The Problem

```java
String name = getUserName();
int length = name.length();  // NullPointerException if name is null
```

Many languages allow variables to be `null`, leading to runtime crashes.

### Solution: Option/Maybe Types

**Rust's Option<T>:**
```rust
fn divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

let result = divide(10, 2);
match result {
    Some(value) => println!("Result: {}", value),
    None => println!("Cannot divide by zero"),
}

// Or use combinators
let result = divide(10, 2).unwrap_or(0);  // Default to 0 if None
```

**Java's Optional<T>:**
```java
import java.util.Optional;

public Optional<String> findUserName(int id) {
    if (id == 1) {
        return Optional.of("Alice");
    } else {
        return Optional.empty();
    }
}

Optional<String> name = findUserName(1);
name.ifPresent(n -> System.out.println("Name: " + n));

String result = name.orElse("Unknown");  // Default value
```

**TypeScript:**
```javascript
function divide(a: number, b: number): number | null {
    return b === 0 ? null : a / b;
}

let result = divide(10, 2);
if (result !== null) {
    console.log("Result:", result);
} else {
    console.log("Cannot divide by zero");
}
```

**Benefits**: Forces you to handle the absence of a value explicitly. No more `NullPointerException` surprises.

---

## Type Annotations and Documentation

Even in dynamically-typed languages, you can (and should) document types.

### Python Type Hints

```python
def greet(name: str) -> str:
    """
    Greet a person by name.

    Args:
        name: The person's name

    Returns:
        A greeting message
    """
    return f"Hello, {name}!"

# Type checker (mypy) can catch errors:
# greet(42)  # Error: Argument 1 has incompatible type "int"; expected "str"
```

### JavaScript with JSDoc

```javascript
/**
 * Calculate the area of a rectangle
 * @param {number} width - The width
 * @param {number} height - The height
 * @returns {number} The area
 */
function area(width, height) {
    return width * height;
}
```

### TypeScript

```javascript
function area(width: number, height: number): number {
    return width * height;
}

// area("5", 10);  // Error: Argument of type 'string' is not assignable to 'number'
```

---

## Exercises

### Exercise 1: Type System Analysis

Given this JavaScript code:
```javascript
let x = "10";
let y = 5;
console.log(x + y);  // "105"
console.log(x - y);  // 5
```

1. Why does `+` produce `"105"` but `-` produces `5`?
2. Would this work in Python? Why or why not?
3. Is this strong or weak typing? Static or dynamic?

### Exercise 2: Implement a Stack ADT

Implement a stack ADT in your language of choice:
- Use an array as the underlying data structure
- Implement `push`, `pop`, `peek`, `is_empty`
- Test with integers, then test with strings (same code should work)

### Exercise 3: Generics

Implement a generic `Pair<T, U>` class that holds two values of potentially different types:
- Constructor: `Pair(T first, U second)`
- Methods: `getFirst()`, `getSecond()`, `setFirst(T)`, `setSecond(U)`
- Test: `Pair<String, Integer>` for ("Alice", 30)

### Exercise 4: Option Type

Implement a simple `Option<T>` type (similar to Rust or Java):
- `Some(value)`: Contains a value
- `None`: Empty
- Methods:
  - `isSome()`: Returns true if Some
  - `isNone()`: Returns true if None
  - `unwrap()`: Returns value or throws error if None
  - `unwrapOr(default)`: Returns value or default if None

### Exercise 5: ADT Design

Design an ADT for a **Library System**:
- What operations should it support?
  - Add book
  - Remove book
  - Search by title, author, ISBN
  - Borrow book
  - Return book
- Define the interface (don't implement yet)
- Consider: What data structures could implement this?

### Exercise 6: Null Safety

Refactor this Java code to use `Optional`:

```java
public String getUserEmail(int userId) {
    if (userId == 1) {
        return "alice@example.com";
    }
    return null;
}

String email = getUserEmail(1);
System.out.println(email.toUpperCase());  // Potential NullPointerException
```

Make it safe using `Optional<String>`.

---

## Summary

- **Types classify data**: primitives (int, float, bool, char, string), composites (arrays, structs, tuples)
- **Type systems**:
  - **Static vs dynamic**: compile-time vs runtime checking
  - **Strong vs weak**: strict vs permissive type rules
  - **Type inference**: automatic type deduction
- **Abstract Data Types**: Interface (what) vs implementation (how)
  - Stack, Queue, Map/Dictionary
- **Generics/Templates**: Type-parameterized code for reusability and type safety
- **Algebraic Data Types**: Sum types (enums), product types (tuples/records)
- **Null safety**: `Option`/`Maybe`/`Optional` types prevent null-related crashes
- **Documentation**: Type annotations make code clearer and catch errors early

**Key Insight**: Abstraction is about managing complexity. ADTs let you think at a higher level without worrying about implementation details. Types help catch errors and make code more maintainable.

---

**Previous**: [Programming Paradigms](./02_Programming_Paradigms.md) | **Next**: [Control Flow Patterns](./04_Control_Flow_Patterns.md)
