# 03. Ownership

**Previous**: [Variables and Types](./02_Variables_and_Types.md) | **Next**: [Borrowing and References](./04_Borrowing_and_References.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the three ownership rules and why Rust enforces them
2. Distinguish between stack allocation and heap allocation
3. Trace how move semantics transfer ownership of heap-allocated data
4. Identify when the `Copy` trait is used instead of a move
5. Implement `Clone` for explicit deep copies

---

Ownership is Rust's central innovation — a compile-time system that guarantees memory safety without a garbage collector. Every other Rust concept (borrowing, lifetimes, smart pointers) is built on top of these three rules:

1. **Each value has exactly one owner.**
2. **There can only be one owner at a time.**
3. **When the owner goes out of scope, the value is dropped.**

## Table of Contents
1. [Stack vs Heap](#1-stack-vs-heap)
2. [The Three Ownership Rules](#2-the-three-ownership-rules)
3. [Move Semantics](#3-move-semantics)
4. [Copy and Clone](#4-copy-and-clone)
5. [Ownership and Functions](#5-ownership-and-functions)
6. [Practice Problems](#6-practice-problems)

---

## 1. Stack vs Heap

Understanding where data lives is essential for understanding ownership.

```
STACK (fast, fixed-size)              HEAP (flexible, dynamic-size)
┌──────────────────────┐              ┌──────────────────────────┐
│ Push/pop operations   │              │ Allocator finds free     │
│ Last-in, first-out    │              │ space, returns pointer   │
│ Size known at compile │              │ Size determined at       │
│ time                  │              │ runtime                  │
│                       │  pointer     │                          │
│  x: i32 = 42         │──────────┐   │                          │
│  ptr ────────────────────────── │──>│  "hello world" (bytes)   │
│  len: 11              │         │   │                          │
│  capacity: 11         │         │   │                          │
└──────────────────────┘              └──────────────────────────┘
```

| Property | Stack | Heap |
|----------|-------|------|
| Speed | Very fast (pointer bump) | Slower (allocator search) |
| Size | Known at compile time | Determined at runtime |
| Access | Direct | Via pointer (indirection) |
| Examples | `i32`, `f64`, `bool`, `[i32; 5]` | `String`, `Vec<T>`, `Box<T>` |
| Cleanup | Automatic (scope exit) | Requires `drop` (Rust) or GC/manual free |

---

## 2. The Three Ownership Rules

### Rule 1: Each value has exactly one owner

```rust
fn main() {
    let s = String::from("hello");  // s owns the String
    // The String data lives on the heap
    // The variable s (on the stack) holds: pointer, length, capacity
}
```

### Rule 2: One owner at a time

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;  // Ownership MOVES from s1 to s2
    // println!("{s1}");  // ERROR: s1 is no longer valid
    println!("{s2}");     // OK: s2 is the owner now
}
```

### Rule 3: Value is dropped when owner goes out of scope

```rust
fn main() {
    {
        let s = String::from("hello");
        // s is valid here
    }
    // s is out of scope — Rust calls drop(s), freeing heap memory
    // This is like C++'s RAII (Resource Acquisition Is Initialization)
}
```

---

## 3. Move Semantics

When you assign a heap-allocated value to another variable, Rust **moves** it — the original variable becomes invalid:

```rust
fn main() {
    let s1 = String::from("hello");

    // BEFORE move:
    // Stack:                    Heap:
    // s1: [ptr, len=5, cap=5] → "hello"

    let s2 = s1;

    // AFTER move:
    // Stack:                    Heap:
    // s1: [INVALID]
    // s2: [ptr, len=5, cap=5] → "hello"

    // Why not copy? Because two owners would mean double-free:
    // If both s1 and s2 owned the same heap data, dropping both
    // would free the same memory twice → undefined behavior.
    // Move prevents this at compile time.
}
```

This diagram shows what happens in memory:

```
Before: let s1 = String::from("hello");

  s1 ─────────┐
  [ptr|5|5]   │
              ▼
         ┌─────────┐
         │ h e l l o│
         └─────────┘

After: let s2 = s1;

  s1 ─────────╳  (invalidated)
  s2 ─────────┐
  [ptr|5|5]   │
              ▼
         ┌─────────┐
         │ h e l l o│
         └─────────┘
```

---

## 4. Copy and Clone

### 4.1 Copy Trait — Implicit Bitwise Copy

Types that implement `Copy` are duplicated on assignment instead of moved. These are small, stack-only types where copying is cheap:

```rust
fn main() {
    let x: i32 = 42;
    let y = x;      // Copy, not move — x is still valid
    println!("x={x}, y={y}");  // x=42, y=42

    // Types that implement Copy:
    // - All integer types (i8, u8, i32, u64, etc.)
    // - Floating-point types (f32, f64)
    // - bool
    // - char
    // - Tuples of Copy types: (i32, f64) is Copy, (i32, String) is NOT
    // - Fixed-size arrays of Copy types: [i32; 5] is Copy
}
```

A type can implement `Copy` only if all of its fields are `Copy` **and** it does not implement `Drop` (the custom destructor trait).

### 4.2 Clone Trait — Explicit Deep Copy

For heap-allocated types, use `.clone()` to make an explicit deep copy:

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();  // Deep copy — new heap allocation
    println!("s1={s1}, s2={s2}");  // Both valid

    // After clone:
    // s1 → heap: "hello"  (original)
    // s2 → heap: "hello"  (copy — different allocation)
}
```

> **Guideline**: Prefer borrowing (Lesson 04) over cloning. Clone when you genuinely need independent ownership. Excessive cloning defeats Rust's zero-cost philosophy.

### 4.3 Deriving Copy and Clone

```rust
// Your own types can be Copy if all fields are Copy
#[derive(Debug, Copy, Clone)]
struct Point {
    x: f64,
    y: f64,
}

// This CANNOT be Copy because String is not Copy
#[derive(Debug, Clone)]
struct Person {
    name: String,  // Heap-allocated → no Copy
    age: u32,
}

fn main() {
    let p1 = Point { x: 1.0, y: 2.0 };
    let p2 = p1;  // Copy — p1 is still valid
    println!("{p1:?}");

    let alice = Person { name: String::from("Alice"), age: 30 };
    let bob = alice.clone();  // Must explicitly clone
    // let bob = alice;       // This would MOVE, invalidating alice
    println!("{alice:?}");
}
```

---

## 5. Ownership and Functions

### 5.1 Passing Ownership to Functions

```rust
fn take_ownership(s: String) {
    println!("Got: {s}");
}   // s is dropped here — heap memory freed

fn make_copy(n: i32) {
    println!("Got: {n}");
}   // n is dropped, but it was just a stack copy

fn main() {
    let greeting = String::from("hello");
    take_ownership(greeting);
    // println!("{greeting}");  // ERROR: greeting was moved into the function

    let num = 42;
    make_copy(num);
    println!("{num}");  // OK: i32 implements Copy
}
```

### 5.2 Returning Ownership

```rust
fn create_string() -> String {
    let s = String::from("created");
    s   // Ownership moves to the caller
}

fn take_and_give_back(s: String) -> String {
    println!("Borrowed briefly: {s}");
    s   // Return ownership to caller
}

fn main() {
    let s1 = create_string();       // s1 owns "created"
    let s2 = take_and_give_back(s1); // s1 → function → s2
    println!("{s2}");
}
```

> Passing ownership in and out is cumbersome. The next lesson introduces **borrowing** — a way to let functions access data without taking ownership.

### 5.3 Returning Multiple Values

```rust
fn calculate_length(s: String) -> (String, usize) {
    let length = s.len();
    (s, length)  // Return both the String AND the length
}

fn main() {
    let s = String::from("hello");
    let (s, len) = calculate_length(s);
    println!("'{s}' has length {len}");
}
```

---

## 6. Practice Problems

### Exercise 1: Move Prediction
Without running the code, predict which `println!` statements will compile and which will cause errors. Then verify with `cargo check`.

```rust
fn main() {
    let a = String::from("hello");
    let b = a;
    let c = b;
    println!("{a}");  // ?
    println!("{b}");  // ?
    println!("{c}");  // ?
}
```

### Exercise 2: Copy vs Move
Write a function `double(x: i32) -> i32` that returns `x * 2`. Call it with a variable and verify the original is still usable. Then write `exclaim(s: String) -> String` that appends "!" — show that you must capture the return value to keep using the string.

### Exercise 3: Ownership Transfer
Write a function `first_word(s: String) -> (String, String)` that takes ownership of a String, extracts the first word, and returns both the first word and the remaining text. Demonstrate usage in `main`.

### Exercise 4: Clone Tradeoff
Given a `Vec<String>` with 1000 elements, explain the performance difference between `let v2 = v1;` (move) and `let v2 = v1.clone();` (deep copy). When would cloning be justified?

### Exercise 5: Custom Copy Type
Define a `Color` struct with `r: u8, g: u8, b: u8` that derives `Copy` and `Clone`. Write a function that takes a `Color` by value and verify the original is still accessible after the call.

---

## References
- [The Rust Book — Ownership](https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html)
- [Rust by Example — Ownership](https://doc.rust-lang.org/rust-by-example/scope/move.html)

---

**Previous**: [Variables and Types](./02_Variables_and_Types.md) | **Next**: [Borrowing and References](./04_Borrowing_and_References.md)
