# 04. Borrowing and References

**Previous**: [Ownership](./03_Ownership.md) | **Next**: [Slices](./05_Slices.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create immutable (`&T`) and mutable (`&mut T`) references
2. State the two borrowing rules and explain why they prevent data races
3. Identify dangling reference errors and fix them
4. Choose between passing by reference, by value, and by mutable reference

---

The previous lesson showed that passing ownership into functions is cumbersome — you must return values to regain access. **Borrowing** solves this: you can let a function *read* or *modify* data through a reference without transferring ownership. The compiler enforces two rules that eliminate data races at compile time.

## Table of Contents
1. [References and Borrowing](#1-references-and-borrowing)
2. [The Two Borrowing Rules](#2-the-two-borrowing-rules)
3. [Mutable References](#3-mutable-references)
4. [Dangling References](#4-dangling-references)
5. [Reference Patterns](#5-reference-patterns)
6. [Practice Problems](#6-practice-problems)

---

## 1. References and Borrowing

A **reference** is a pointer that borrows a value without owning it. The value it points to will not be dropped when the reference goes out of scope.

```rust
fn calculate_length(s: &String) -> usize {
    s.len()
}   // s goes out of scope, but it doesn't own the String — nothing is dropped

fn main() {
    let s1 = String::from("hello");
    let len = calculate_length(&s1);  // &s1 creates a reference
    println!("'{s1}' has length {len}");  // s1 is still valid!
}
```

```
Ownership vs Borrowing:

  Ownership (move):                 Borrowing (reference):
  main → fn: s moved in            main → fn: &s lent temporarily
  main loses access                 main keeps ownership
  fn must return to give back       fn returns, reference expires
```

---

## 2. The Two Borrowing Rules

Rust enforces these rules at compile time:

> **Rule 1**: You can have **either** one mutable reference **or** any number of immutable references — but not both at the same time.
>
> **Rule 2**: References must always be **valid** (no dangling pointers).

```rust
fn main() {
    let mut s = String::from("hello");

    // Multiple immutable references — OK
    let r1 = &s;
    let r2 = &s;
    println!("{r1}, {r2}");  // Both valid

    // Mutable reference after immutable ones are done — OK
    // (r1 and r2 are no longer used after this point — NLL)
    let r3 = &mut s;
    r3.push_str(", world");
    println!("{r3}");

    // Simultaneous mutable + immutable — ERROR
    // let r4 = &s;
    // let r5 = &mut s;
    // println!("{r4}, {r5}");  // ERROR: cannot borrow as mutable
}
```

### Why These Rules?

The rules prevent **data races** at compile time. A data race occurs when:
1. Two or more pointers access the same data simultaneously
2. At least one pointer is writing
3. There's no synchronization

Rust's rule (either one writer OR many readers) makes condition 1+2 impossible.

### Non-Lexical Lifetimes (NLL)

The compiler tracks where references are **last used**, not just where they go out of scope:

```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &s;
    let r2 = &s;
    println!("{r1}, {r2}");
    // r1 and r2 are not used after this line — their lifetimes end here

    let r3 = &mut s;   // OK! No conflict because r1, r2 are "dead"
    r3.push_str("!");
    println!("{r3}");
}
```

---

## 3. Mutable References

### 3.1 Basic Mutable Borrowing

```rust
fn append_greeting(s: &mut String) {
    s.push_str(", world!");
}

fn main() {
    let mut s = String::from("hello");  // Variable must be mut
    append_greeting(&mut s);             // Pass mutable reference
    println!("{s}");  // hello, world!
}
```

### 3.2 One Mutable Reference at a Time

```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &mut s;
    // let r2 = &mut s;  // ERROR: cannot borrow s as mutable more than once
    r1.push_str("!");
    println!("{r1}");

    // After r1 is no longer used, we can create a new mutable reference
    let r2 = &mut s;
    r2.push_str("!");
    println!("{r2}");
}
```

### 3.3 Mutable References in Functions

```rust
fn swap_values(a: &mut i32, b: &mut i32) {
    let temp = *a;  // Dereference to get the value
    *a = *b;
    *b = temp;
}

fn main() {
    let mut x = 1;
    let mut y = 2;
    swap_values(&mut x, &mut y);
    println!("x={x}, y={y}");  // x=2, y=1
}
```

---

## 4. Dangling References

A dangling reference points to memory that has been freed. Rust prevents this at compile time:

```rust
// This does NOT compile:
// fn dangle() -> &String {
//     let s = String::from("hello");
//     &s  // ERROR: s is dropped at end of function, reference would dangle
// }

// Solution: return the owned value instead
fn no_dangle() -> String {
    let s = String::from("hello");
    s  // Ownership moves to caller — no dangling reference
}

fn main() {
    let s = no_dangle();
    println!("{s}");
}
```

---

## 5. Reference Patterns

### 5.1 Choosing the Right Parameter Type

```rust
// Takes ownership — caller loses access
fn consume(s: String) { /* ... */ }

// Immutable borrow — caller keeps ownership, function can read
fn inspect(s: &String) { /* ... */ }

// Mutable borrow — caller keeps ownership, function can modify
fn modify(s: &mut String) { /* ... */ }

// General guideline:
// Use &T    when you only need to read
// Use &mut T when you need to modify
// Use T      when you need ownership (storing in a struct, spawning a thread, etc.)
```

### 5.2 The Dereference Operator

```rust
fn main() {
    let x = 5;
    let r = &x;

    // Explicit dereference
    assert_eq!(*r, 5);

    // Rust auto-dereferences in many contexts (dot operator, comparisons)
    let s = String::from("hello");
    let r = &s;
    println!("{}", r.len());  // Auto-deref: same as (*r).len()
}
```

### 5.3 References to References

```rust
fn main() {
    let x = 42;
    let r1 = &x;       // &i32
    let r2 = &r1;      // &&i32
    let r3 = &r2;      // &&&i32

    // Rust auto-dereferences through multiple levels
    assert_eq!(***r3, 42);
    assert_eq!(**r2, 42);
    println!("{r3}");  // 42 — Display auto-derefs
}
```

---

## 6. Practice Problems

### Exercise 1: Fix the Borrow Checker Errors
Fix the following code to compile:

```rust
fn main() {
    let mut s = String::from("hello");
    let r1 = &s;
    let r2 = &mut s;
    println!("{r1}, {r2}");
}
```

### Exercise 2: String Modifier
Write a function `make_uppercase(s: &mut String)` that converts the string to uppercase in-place. Call it from `main` and print the result.

### Exercise 3: Counting Function
Write `count_char(s: &str, c: char) -> usize` that counts occurrences of `c` in `s` without taking ownership. Demonstrate that the original string is still usable after the call.

### Exercise 4: Reference Lifetime
Explain why this function signature is valid but the following implementation is not:

```rust
fn longest(a: &str, b: &str) -> &str {
    if a.len() > b.len() { a } else { b }
}
```

What additional annotation does the compiler require? (Preview of Lesson 11.)

### Exercise 5: Ownership Decision
For each scenario, decide whether a function should take `T`, `&T`, or `&mut T`:
1. A function that prints a user's name
2. A function that adds an item to a shopping cart
3. A function that stores a config object in a global registry
4. A function that computes the hash of a byte slice

---

## References
- [The Rust Book — References and Borrowing](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html)
- [Rust by Example — Borrowing](https://doc.rust-lang.org/rust-by-example/scope/borrow.html)

---

**Previous**: [Ownership](./03_Ownership.md) | **Next**: [Slices](./05_Slices.md)
