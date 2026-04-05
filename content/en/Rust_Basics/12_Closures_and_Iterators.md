# 12. Closures and Iterators

**Previous**: [Lifetimes](./11_Lifetimes.md) | **Next**: [Smart Pointers](./13_Smart_Pointers.md)

**Difficulty**: ⭐⭐⭐

## Learning Objectives

- Define closures using multiple syntax forms and explain how Rust infers their parameter and return types
- Distinguish between `Fn`, `FnMut`, and `FnOnce` traits and predict which trait a given closure implements
- Use `move` closures to transfer ownership of captured variables into the closure
- Chain iterator adaptors (`map`, `filter`, `zip`, etc.) to build declarative data processing pipelines
- Implement the `Iterator` trait for a custom type and explain why Rust iterators are zero-cost abstractions

## Table of Contents

1. [What Are Closures?](#1-what-are-closures)
2. [Closure Syntax](#2-closure-syntax)
3. [Type Inference and Capture Modes](#3-type-inference-and-capture-modes)
4. [Fn, FnMut, and FnOnce Traits](#4-fn-fnmut-and-fnonce-traits)
5. [Move Closures](#5-move-closures)
6. [The Iterator Trait](#6-the-iterator-trait)
7. [Iterator Adaptors](#7-iterator-adaptors)
8. [Consuming Adaptors](#8-consuming-adaptors)
9. [Creating Custom Iterators](#9-creating-custom-iterators)
10. [Performance: Iterators vs Loops](#10-performance-iterators-vs-loops)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. What Are Closures?

A **closure** is an anonymous function that can capture variables from its surrounding scope. Think of closures as "functions with a backpack" — they carry along any outside data they need.

In many languages, closures are called lambdas or anonymous functions. Rust closures are special because the compiler statically determines *how* each variable is captured (by reference, mutable reference, or by value), which means closures are as efficient as hand-written code.

```
Regular function:       fn add(a: i32, b: i32) -> i32 { a + b }
                        ^^ has a name, no capture

Closure:                |a, b| a + b
                        ^^ anonymous, can capture from scope
```

---

## 2. Closure Syntax

Rust offers several closure syntax forms, from concise to verbose:

```rust
fn main() {
    // 1. Single expression — no braces, no type annotations
    let add = |a, b| a + b;
    println!("3 + 4 = {}", add(3, 4));

    // 2. Block body — multiple statements require braces
    let greet = |name: &str| {
        let message = format!("Hello, {}!", name);
        println!("{}", message);
        message // last expression is the return value
    };
    greet("Rustacean");

    // 3. Fully annotated — explicit parameter and return types
    let multiply = |x: i32, y: i32| -> i32 { x * y };
    println!("5 * 6 = {}", multiply(5, 6));

    // 4. No parameters
    let say_hi = || println!("Hi!");
    say_hi();
}
```

Unlike regular functions, closures do **not** require type annotations. The compiler infers types from the first call site. However, once inferred, the types are fixed:

```rust
fn main() {
    let identity = |x| x;

    let s = identity("hello"); // inferred: |&str| -> &str
    // let n = identity(42);   // ERROR: expected &str, found integer
    println!("{}", s);
}
```

---

## 3. Type Inference and Capture Modes

When a closure references a variable from the surrounding scope, Rust decides the **least restrictive** capture mode that still satisfies the closure body:

```
Capture Mode       What Happens              Trait Implemented
─────────────────────────────────────────────────────────────
Immutable borrow   &T                        Fn
Mutable borrow     &mut T                    FnMut
Ownership          T (value moved in)        FnOnce
```

The compiler always picks the lightest mode first:

```rust
fn main() {
    let name = String::from("Alice");

    // Capture by immutable reference — closure only reads `name`
    let greet = || println!("Hello, {name}!");
    greet();
    greet(); // can call multiple times
    println!("name is still accessible: {name}");

    let mut counter = 0;

    // Capture by mutable reference — closure modifies `counter`
    let mut increment = || {
        counter += 1; // needs &mut counter
        println!("counter = {counter}");
    };
    increment();
    increment();
    // `counter` is borrowed mutably while `increment` exists
    // println!("{counter}"); // ERROR if uncommented before last use of `increment`

    let ticket = String::from("VIP-001");

    // Capture by value — closure consumes `ticket`
    let consume = || {
        let _moved = ticket; // takes ownership
        println!("Ticket consumed");
    };
    consume();
    // consume(); // ERROR: closure implements FnOnce, already called
    // println!("{ticket}"); // ERROR: ticket was moved
}
```

---

## 4. Fn, FnMut, and FnOnce Traits

Every closure in Rust implements one or more of three traits, which form a hierarchy:

```
        FnOnce          ← every closure implements this (can be called at least once)
          ▲
          │
        FnMut           ← closures that don't consume captures (can be called multiple times)
          ▲
          │
         Fn             ← closures that don't mutate captures (can be called concurrently)
```

This means: every `Fn` is also `FnMut`, and every `FnMut` is also `FnOnce`.

When you write a function that accepts a closure, choose the most permissive trait that works:

```rust
// Accepts any closure that can be called multiple times without mutation
fn apply_twice<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(f(x))
}

// Accepts closures that may mutate their captured state
fn call_n_times<F: FnMut()>(mut f: F, n: usize) {
    for _ in 0..n {
        f(); // each call may mutate captured variables
    }
}

// Accepts closures that may consume captured values (called exactly once)
fn call_once<F: FnOnce() -> String>(f: F) -> String {
    f() // closure may move out of its captures
}

fn main() {
    // Fn example: pure transformation, no captures modified
    let double = |x| x * 2;
    println!("apply_twice(double, 3) = {}", apply_twice(double, 3)); // 12

    // FnMut example: closure mutates its captured counter
    let mut total = 0;
    call_n_times(|| { total += 1; }, 5);
    println!("total = {total}"); // 5

    // FnOnce example: closure moves a String out
    let greeting = String::from("Hello, world!");
    let result = call_once(|| greeting); // `greeting` moved into return value
    println!("{result}");
}
```

**Guideline**: prefer `Fn` when possible, use `FnMut` when the closure needs to mutate state, and reserve `FnOnce` for closures that consume their captures.

---

## 5. Move Closures

By default, closures borrow variables with the lightest mode. The `move` keyword forces the closure to take **ownership** of all captured variables, regardless of how they are used:

```rust
use std::thread;

fn main() {
    let name = String::from("Alice");

    // Without `move`, this would try to borrow `name`.
    // But threads may outlive the current scope, so Rust requires ownership.
    let handle = thread::spawn(move || {
        // `name` is now owned by this closure — safe to use in another thread
        println!("Hello from thread: {name}");
    });

    // println!("{name}"); // ERROR: `name` was moved into the closure

    handle.join().unwrap();
}
```

For `Copy` types like integers, `move` creates a copy rather than transferring ownership:

```rust
fn main() {
    let x = 42; // i32 implements Copy

    let closure = move || println!("x = {x}");
    closure();

    // x is still usable because i32 was copied, not moved
    println!("x in main = {x}");
}
```

---

## 6. The Iterator Trait

The `Iterator` trait is defined in `std::iter` and requires a single method:

```rust
trait Iterator {
    type Item;                        // the type of each element
    fn next(&mut self) -> Option<Self::Item>; // returns Some(item) or None
}
```

Every call to `next()` advances the iterator by one step. When the sequence is exhausted, it returns `None`. Think of an iterator as a bookmark in a book — each call to `next()` turns one page.

```rust
fn main() {
    let numbers = vec![10, 20, 30];

    // `iter()` borrows elements — yields &i32
    let mut iter = numbers.iter();
    assert_eq!(iter.next(), Some(&10));
    assert_eq!(iter.next(), Some(&20));
    assert_eq!(iter.next(), Some(&30));
    assert_eq!(iter.next(), None); // exhausted

    // Three ways to create iterators from a collection:
    //   iter()      → borrows (&T)
    //   iter_mut()  → borrows mutably (&mut T)
    //   into_iter() → takes ownership (T)

    // `for` loops call `into_iter()` automatically
    for num in &numbers {
        // num is &i32
        print!("{num} ");
    }
    println!();
}
```

The `size_hint()` method provides an optional hint about the remaining length, which collection methods like `collect()` use to pre-allocate memory:

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];
    let iter = v.iter();
    let (lower, upper) = iter.size_hint();
    println!("At least {lower} elements, at most {upper:?} elements");
    // Output: At least 5 elements, at most Some(5) elements
}
```

---

## 7. Iterator Adaptors

Iterator adaptors are **lazy** — they create a new iterator that transforms elements on demand, without doing any work until consumed. You can chain them freely:

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // map: transform each element
    let squares: Vec<i32> = numbers.iter().map(|&x| x * x).collect();
    println!("squares: {squares:?}");

    // filter: keep elements matching a predicate
    let evens: Vec<&i32> = numbers.iter().filter(|&&x| x % 2 == 0).collect();
    println!("evens: {evens:?}");

    // enumerate: attach indices (i, value)
    for (i, val) in numbers.iter().enumerate() {
        if i < 3 {
            print!("[{i}]={val} ");
        }
    }
    println!();

    // zip: pair up two iterators element by element
    let names = vec!["Alice", "Bob", "Charlie"];
    let scores = vec![95, 87, 92];
    let results: Vec<_> = names.iter().zip(scores.iter()).collect();
    println!("results: {results:?}");

    // take and skip: slice the iterator
    let first_three: Vec<&i32> = numbers.iter().take(3).collect();
    let after_seven: Vec<&i32> = numbers.iter().skip(7).collect();
    println!("first 3: {first_three:?}, after 7: {after_seven:?}");

    // chain: concatenate two iterators
    let a = vec![1, 2, 3];
    let b = vec![4, 5, 6];
    let combined: Vec<&i32> = a.iter().chain(b.iter()).collect();
    println!("chained: {combined:?}");
}
```

**Key insight**: because adaptors are lazy, this builds a pipeline description. No work happens until a consuming adaptor (like `collect`) drives the iteration:

```
numbers.iter()        → [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  .filter(|x| even)  → [   2,    4,    6,    8,    10 ]    ← lazy
  .map(|x| x * x)    → [   4,   16,   36,   64,   100]    ← lazy
  .collect()          → Vec[4, 16, 36, 64, 100]            ← drives iteration
```

---

## 8. Consuming Adaptors

Consuming adaptors call `next()` repeatedly and produce a final result:

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5];

    // sum: add all elements
    let total: i32 = numbers.iter().sum();
    println!("sum = {total}"); // 15

    // fold: generalized reduction with an accumulator
    let product = numbers.iter().fold(1, |acc, &x| acc * x);
    println!("product = {product}"); // 120

    // any: does at least one element satisfy the predicate?
    let has_even = numbers.iter().any(|&x| x % 2 == 0);
    println!("has even? {has_even}"); // true

    // all: do all elements satisfy the predicate?
    let all_positive = numbers.iter().all(|&x| x > 0);
    println!("all positive? {all_positive}"); // true

    // find: first element matching a predicate (returns Option)
    let first_even = numbers.iter().find(|&&x| x % 2 == 0);
    println!("first even = {first_even:?}"); // Some(2)

    // position: index of first match (returns Option<usize>)
    let pos = numbers.iter().position(|&x| x == 3);
    println!("position of 3 = {pos:?}"); // Some(2)

    // collect with type turbofish — gather into different collections
    let as_vec: Vec<i32> = (1..=5).collect();
    let as_string: String = vec!['R', 'u', 's', 't'].into_iter().collect();
    println!("{as_vec:?}, {as_string}");
}
```

A powerful pattern combines adaptors and consumers in a single pipeline:

```rust
fn main() {
    let text = "hello world, hello rust, goodbye world";

    // Count how many words start with 'h'
    let h_count = text.split_whitespace()
        .filter(|word| word.starts_with('h'))
        .count();
    println!("Words starting with 'h': {h_count}"); // 2

    // Build a comma-separated string of uppercase words longer than 4 chars
    let result: String = text.split_whitespace()
        .filter(|w| w.len() > 4)
        .map(|w| w.to_uppercase())
        .collect::<Vec<_>>()
        .join(", ");
    println!("{result}"); // HELLO, WORLD,, HELLO, RUST,, GOODBYE, WORLD
}
```

---

## 9. Creating Custom Iterators

To make any type iterable, implement the `Iterator` trait. Here is a counter that generates a Fibonacci-like sequence:

```rust
struct Fibonacci {
    a: u64,
    b: u64,
}

impl Fibonacci {
    fn new() -> Self {
        Fibonacci { a: 0, b: 1 }
    }
}

impl Iterator for Fibonacci {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.a;
        let new_next = self.a + self.b;
        self.a = self.b;
        self.b = new_next;
        Some(current) // infinite iterator — never returns None
    }
}

fn main() {
    // Take the first 10 Fibonacci numbers
    let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
    println!("First 10 Fibonacci: {fibs:?}");
    // [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    // Sum of Fibonacci numbers below 100
    let sum: u64 = Fibonacci::new()
        .take_while(|&n| n < 100)
        .sum();
    println!("Sum of Fibs below 100: {sum}");

    // Combine with other adaptors seamlessly
    let even_fibs: Vec<u64> = Fibonacci::new()
        .take(20)
        .filter(|n| n % 2 == 0)
        .collect();
    println!("Even Fibonacci (first 20): {even_fibs:?}");
}
```

You can also implement `IntoIterator` for a collection type, which enables `for` loops:

```rust
struct Countdown {
    remaining: u32,
}

impl Countdown {
    fn from(start: u32) -> Self {
        Countdown { remaining: start }
    }
}

impl Iterator for Countdown {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None // signal end of iteration
        } else {
            self.remaining -= 1;
            Some(self.remaining + 1) // return value before decrement
        }
    }
}

fn main() {
    // Use directly in a for loop (Iterator automatically provides IntoIterator)
    for n in Countdown::from(5) {
        print!("{n}... ");
    }
    println!("Liftoff!");
    // 5... 4... 3... 2... 1... Liftoff!
}
```

---

## 10. Performance: Iterators vs Loops

A common concern is whether iterator chains add overhead. Rust's answer is **zero-cost abstractions** — the compiler optimizes iterator chains into the same machine code as hand-written loops.

```rust
fn sum_of_squares_loop(numbers: &[i32]) -> i32 {
    let mut total = 0;
    for &n in numbers {
        if n % 2 == 0 {
            total += n * n;
        }
    }
    total
}

fn sum_of_squares_iter(numbers: &[i32]) -> i32 {
    numbers.iter()
        .filter(|&&n| n % 2 == 0)
        .map(|&n| n * n)
        .sum()
}

fn main() {
    let data: Vec<i32> = (1..=1000).collect();

    let result_loop = sum_of_squares_loop(&data);
    let result_iter = sum_of_squares_iter(&data);
    assert_eq!(result_loop, result_iter);

    println!("Sum of squares of even numbers 1..=1000: {result_loop}");
}
```

Both functions compile to virtually identical assembly. The iterator version is often **preferred** because:

- It eliminates the possibility of off-by-one index errors
- The intent is clearer (declarative vs imperative)
- The compiler can apply SIMD auto-vectorization more reliably

The Rust documentation calls iterators "one of Rust's zero-cost abstractions" — you pay no runtime penalty for the higher-level abstraction.

---

## 11. Practice Problems

### Problem 1: Word Frequency Counter

Write a function `word_frequencies(text: &str) -> Vec<(String, usize)>` that splits text into lowercase words, counts how often each word appears, and returns the counts sorted by frequency (highest first). Use iterator adaptors and a `HashMap`.

### Problem 2: Custom Range Iterator

Create a `StepRange` struct that iterates from `start` to `end` (exclusive) with a custom `step` size. Implement the `Iterator` trait. For example, `StepRange::new(0, 20, 3)` should yield `0, 3, 6, 9, 12, 15, 18`.

### Problem 3: Closure-Based Event System

Design a simple event system where you can register handler closures and dispatch events. Create a `Dispatcher` struct with methods `on(event_name, callback)` and `emit(event_name, data)`. Decide which `Fn` trait the callbacks should implement and explain why.

### Problem 4: Parallel Pipeline

Using only iterator adaptors and consumers, write a single expression that:
1. Takes a `Vec<String>` of lines from a CSV-like format (`"name,score"`)
2. Parses each line into a `(String, u32)` tuple
3. Filters out entries with score below 50
4. Computes the average score of the remaining entries

### Problem 5: Infinite Iterator Composition

Create two custom infinite iterators: `Naturals` (1, 2, 3, ...) and `Powers` (which takes a base and yields base^0, base^1, base^2, ...). Then compose them with `zip` and `take_while` to find all pairs `(n, 2^n)` where `2^n < 1_000_000`.

---

## 12. References

- [The Rust Programming Language, Ch. 13: Functional Language Features](https://doc.rust-lang.org/book/ch13-00-functional-features.html)
- [Rust by Example: Closures](https://doc.rust-lang.org/rust-by-example/fn/closures.html)
- [Rust by Example: Iterators](https://doc.rust-lang.org/rust-by-example/trait/iter.html)
- [std::iter Module Documentation](https://doc.rust-lang.org/std/iter/index.html)
- [Iterator trait API Reference](https://doc.rust-lang.org/std/iter/trait.Iterator.html)

---

**Previous**: [Lifetimes](./11_Lifetimes.md) | **Next**: [Smart Pointers](./13_Smart_Pointers.md)
