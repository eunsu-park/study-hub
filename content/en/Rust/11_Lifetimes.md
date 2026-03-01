# 11. Lifetimes

**Previous**: [Traits and Generics](./10_Traits_and_Generics.md) | **Next**: [Closures and Iterators](./12_Closures_and_Iterators.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what lifetimes are and why Rust needs them to prevent dangling references
2. Annotate function signatures and struct definitions with lifetime parameters
3. Apply the three lifetime elision rules to determine when explicit annotations are unnecessary
4. Use `'static` lifetime correctly for string literals and owned data
5. Identify common lifetime patterns and avoid anti-patterns that fight the borrow checker

---

Lifetimes are Rust's answer to a fundamental question: **how long does a reference stay valid?** In C, you can return a pointer to a local variable and crash at runtime. In garbage-collected languages, the runtime keeps everything alive until no references remain. Rust takes a third path: the compiler statically verifies that every reference outlives its use, with zero runtime cost.

If ownership is about "who owns this data?", lifetimes are about "how long does this borrowed view of the data last?" Think of it like a library book: ownership is who bought the book, and the lifetime is the due date on the library card. The compiler is the librarian who refuses to let you check out a book that is already overdue.

Lifetimes are the most conceptually challenging part of Rust. Take your time — once they click, the borrow checker transforms from an adversary into an ally.

## Table of Contents
1. [Why Lifetimes Exist](#1-why-lifetimes-exist)
2. [Lifetime Annotation Syntax](#2-lifetime-annotation-syntax)
3. [Lifetimes in Function Signatures](#3-lifetimes-in-function-signatures)
4. [The Classic Example: longest()](#4-the-classic-example-longest)
5. [Lifetime Elision Rules](#5-lifetime-elision-rules)
6. [Lifetimes in Struct Definitions](#6-lifetimes-in-struct-definitions)
7. [The 'static Lifetime](#7-the-static-lifetime)
8. [Lifetime Bounds on Generic Types](#8-lifetime-bounds-on-generic-types)
9. [Common Patterns and Anti-Patterns](#9-common-patterns-and-anti-patterns)
10. [Practice Problems](#10-practice-problems)

---

## 1. Why Lifetimes Exist

Lifetimes prevent **dangling references** — pointers to memory that has been freed:

```rust
fn main() {
    let r;                // declare r (no value yet)
    {
        let x = 5;
        r = &x;          // borrow x
    }                     // x is dropped here
    // println!("{}", r); // ERROR: r is a dangling reference to dropped x
}
```

The compiler catches this because the lifetime of `x` is shorter than the lifetime of `r`:

```
Scope diagram:

    let r;              ─────────────────────────── r lives here
    {                       │
        let x = 5;         ├── x lives here
        r = &x;            │   r borrows x
    }                       │   x dropped ← PROBLEM: r still exists
                        ────┘
    // r would be dangling
```

Without lifetime checking, this would compile and produce undefined behavior (like in C). Rust refuses to compile it.

---

## 2. Lifetime Annotation Syntax

Lifetime annotations do **not** change how long any value lives. They describe the **relationships** between lifetimes of references, so the compiler can verify safety:

```rust
&i32        // a reference (implicit lifetime)
&'a i32     // a reference with explicit lifetime 'a
&'a mut i32 // a mutable reference with explicit lifetime 'a
```

The `'a` (pronounced "tick a") is a **lifetime parameter** — a name for a scope. By convention, lifetimes use short lowercase names: `'a`, `'b`, `'c`.

**Key insight:** You never create lifetimes. You only label them so the compiler can check that the relationships make sense. It is like putting name tags on scopes that already exist.

---

## 3. Lifetimes in Function Signatures

When a function takes references and returns a reference, the compiler needs to know how the output's lifetime relates to the inputs':

```rust
// This won't compile — the compiler doesn't know which input's
// lifetime the return value is tied to
// fn first_word(s: &str) -> &str { ... }
// Actually, this DOES compile due to elision rules (covered in Section 5).
// But let's see a case where elision doesn't help:

// Two input references — which one does the return value borrow from?
// fn pick(a: &str, b: &str) -> &str { ... }
// ERROR: missing lifetime specifier

// Solution: annotate to express the relationship
fn pick<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.len() >= b.len() { a } else { b }
}

fn main() {
    let s1 = String::from("long string");
    let result;
    {
        let s2 = String::from("short");
        result = pick(&s1, &s2);
        println!("result: {}", result); // OK: both s1 and s2 are alive
    }
    // println!("{}", result); // ERROR if uncommented:
    // s2 was dropped, and result might reference s2
}
```

What the annotation `'a` means here: "the returned reference will be valid for as long as **both** input references are valid." The compiler uses this to ensure you do not use the result after either input is dropped.

---

## 4. The Classic Example: longest()

This is the canonical lifetime teaching example from The Rust Book:

```rust
// Returns a reference to the longer of two string slices
// 'a means: the returned reference lives at least as long as
// the shorter of the two input lifetimes
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("abcdef");

    // Case 1: both references live long enough ✓
    {
        let string2 = String::from("xyz");
        let result = longest(string1.as_str(), string2.as_str());
        println!("Longest: {}", result); // OK
    }

    // Case 2: result tries to outlive one of the inputs ✗
    // let result;
    // {
    //     let string2 = String::from("xyz");
    //     result = longest(string1.as_str(), string2.as_str());
    // }
    // println!("Longest: {}", result); // ERROR: string2 doesn't live long enough
}
```

### 4.1 What 'a Actually Represents

```
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str

The lifetime 'a is the OVERLAP (intersection) of x's and y's lifetimes:

    string1: ──────────────────────────────────
    string2:        ─────────────────
                    ▲               ▲
                    │    'a = this  │
                    │    overlap    │
                    └───────────────┘

The return value is valid only within this overlap region.
```

### 4.2 When You Don't Need Lifetime Annotations

If the function only borrows from one input, the compiler can figure it out:

```rust
// Only one input reference → compiler knows the output borrows from it
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b' ' {
            return &s[..i];
        }
    }
    s
}

// Returning an owned value → no borrow relationship at all
fn make_greeting(name: &str) -> String {
    format!("Hello, {}!", name) // returns an owned String, not a reference
}
```

---

## 5. Lifetime Elision Rules

Early Rust required lifetime annotations everywhere. That was tedious, so the compiler learned three **elision rules** — patterns so common that the compiler applies them automatically:

```
Rule 1: Each input reference gets its own lifetime parameter.

    fn foo(x: &str)            → fn foo<'a>(x: &'a str)
    fn foo(x: &str, y: &str)   → fn foo<'a, 'b>(x: &'a str, y: &'b str)

Rule 2: If there is exactly ONE input lifetime, it is assigned
         to ALL output references.

    fn foo(x: &str) -> &str    → fn foo<'a>(x: &'a str) -> &'a str
    (This is why first_word works without annotations!)

Rule 3: If one of the inputs is &self or &mut self, the lifetime
         of self is assigned to all output references.

    impl Foo {
        fn bar(&self, x: &str) -> &str
        // becomes:
        fn bar<'a, 'b>(&'a self, x: &'b str) -> &'a str
        // output borrows from self, not from x
    }
```

### 5.1 Applying the Rules Step by Step

```
Example: fn longest(x: &str, y: &str) -> &str

Step 1 (Rule 1): fn longest<'a, 'b>(x: &'a str, y: &'b str) -> &str
Step 2 (Rule 2): TWO input lifetimes → Rule 2 does not apply
Step 3 (Rule 3): No &self → Rule 3 does not apply

Result: output lifetime is STILL ambiguous → compiler error!
         You MUST add explicit annotations.
```

```
Example: fn first_word(s: &str) -> &str

Step 1 (Rule 1): fn first_word<'a>(s: &'a str) -> &str
Step 2 (Rule 2): ONE input lifetime → assign to output
                  fn first_word<'a>(s: &'a str) -> &'a str

Result: fully resolved → no annotations needed!
```

---

## 6. Lifetimes in Struct Definitions

When a struct holds references, it needs lifetime annotations to ensure the referenced data outlives the struct:

```rust
// ImportantExcerpt borrows from a string — it cannot outlive that string
#[derive(Debug)]
struct ImportantExcerpt<'a> {
    part: &'a str,
}

impl<'a> ImportantExcerpt<'a> {
    // Rule 3 applies: output borrows from &self
    fn level(&self) -> i32 {
        3
    }

    // Rule 3: returned &str borrows from &self (not from announcement)
    fn announce_and_return(&self, announcement: &str) -> &str {
        println!("Attention: {}", announcement);
        self.part
    }
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");

    // The excerpt borrows from novel — it must not outlive novel
    let first_sentence;
    {
        let i = novel.find('.').unwrap_or(novel.len());
        first_sentence = ImportantExcerpt {
            part: &novel[..i],
        };
    }
    // first_sentence is still valid because novel is still alive
    println!("{:?}", first_sentence);
}
```

### 6.1 Struct with Multiple Lifetimes

```rust
// Sometimes fields borrow from different sources
#[derive(Debug)]
struct Highlight<'text, 'query> {
    text: &'text str,
    query: &'query str,
}

impl<'text, 'query> Highlight<'text, 'query> {
    fn new(text: &'text str, query: &'query str) -> Self {
        Highlight { text, query }
    }

    fn matches(&self) -> bool {
        self.text.contains(self.query)
    }
}

fn main() {
    let document = String::from("Rust is a systems programming language");
    let search = String::from("systems");

    let highlight = Highlight::new(&document, &search);
    println!("Match: {}", highlight.matches()); // true
}
```

---

## 7. The 'static Lifetime

`'static` is the longest possible lifetime — the entire duration of the program:

```rust
fn main() {
    // String literals are 'static — they're embedded in the binary
    let s: &'static str = "I live forever";

    // Type-level constants are also 'static
    static GREETING: &str = "Hello, world!";
    println!("{}", GREETING);
}
```

### 7.1 'static Does Not Always Mean "Literal"

Owned types satisfy `'static` bounds because they don't borrow from anything:

```rust
use std::fmt::Display;

// T: Display + 'static means T implements Display AND
// does not contain any non-'static references
fn print_static<T: Display + 'static>(item: T) {
    println!("{}", item);
}

fn main() {
    print_static(42);                    // i32 is 'static (no references)
    print_static(String::from("hello")); // String is 'static (owns its data)
    // print_static(&String::from("hello")); // temporary reference — NOT 'static

    let s = String::from("hello");
    // print_static(&s); // &String has lifetime of s, which is not 'static
}
```

### 7.2 When You See 'static in Error Messages

A common source of confusion:

```
error: `x` does not live long enough
  --> src/main.rs:5:5
   |
   = note: ...borrowed value must be valid for the static lifetime...
```

This usually means you are trying to store a reference where an owned value is expected. The fix is almost always to **clone or own the data**, not to add `'static`:

```rust
// BAD: fighting the compiler with 'static
// fn get_name() -> &'static str {
//     let name = String::from("Rust");
//     &name // ERROR: returning reference to local
// }

// GOOD: return an owned String
fn get_name() -> String {
    String::from("Rust")
}
```

---

## 8. Lifetime Bounds on Generic Types

You can combine lifetimes with generics and trait bounds:

```rust
use std::fmt::Display;

// T must outlive 'a AND implement Display
fn announce<'a, T: Display>(text: &'a str, extra: T) -> &'a str {
    println!("Announcement: {}", extra);
    text
}

// Lifetime bound on a generic: T: 'a means "T outlives 'a"
// This is needed when T might contain references
struct Wrapper<'a, T: 'a> {
    value: &'a T,
}

fn main() {
    let x = 42;
    let w = Wrapper { value: &x };
    println!("wrapped: {}", w.value);

    let text = String::from("hello");
    let result = announce(&text, 42);
    println!("{}", result);
}
```

### 8.1 Lifetime Bounds in Practice

```rust
// A cache that borrows its data source
struct Cache<'src, T>
where
    T: AsRef<str> + 'src, // T must outlive 'src and be convertible to &str
{
    source: &'src T,
    cached_len: usize,
}

impl<'src, T: AsRef<str> + 'src> Cache<'src, T> {
    fn new(source: &'src T) -> Self {
        let len = source.as_ref().len();
        Cache {
            source,
            cached_len: len,
        }
    }

    fn get(&self) -> &str {
        self.source.as_ref()
    }
}

fn main() {
    let data = String::from("Hello, lifetime bounds!");
    let cache = Cache::new(&data);
    println!("Cached ({} bytes): {}", cache.cached_len, cache.get());
}
```

---

## 9. Common Patterns and Anti-Patterns

### 9.1 Pattern: Return Owned Data Instead of References

When lifetime annotations get complicated, ask yourself: "Can I just return an owned value?"

```rust
// Complex: lifetime annotations, borrow checker gymnastics
fn get_display_name<'a>(first: &'a str, last: &'a str, use_full: bool) -> &'a str {
    if use_full {
        // PROBLEM: can't return a newly created string as a reference
        // let full = format!("{} {}", first, last);
        // &full  // ERROR: returning reference to local variable
        first // workaround — but not what we wanted
    } else {
        first
    }
}

// Simple: just return a String
fn get_display_name_owned(first: &str, last: &str, use_full: bool) -> String {
    if use_full {
        format!("{} {}", first, last) // no lifetime issues!
    } else {
        first.to_string()
    }
}
```

### 9.2 Pattern: Struct Borrowing from Its Constructor Argument

```rust
// Good pattern: struct borrows data that outlives it
struct Parser<'input> {
    input: &'input str,
    position: usize,
}

impl<'input> Parser<'input> {
    fn new(input: &'input str) -> Self {
        Parser { input, position: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.position..].chars().next()
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek() {
            self.position += c.len_utf8();
        }
    }

    fn remaining(&self) -> &'input str {
        &self.input[self.position..]
    }
}

fn main() {
    let source = String::from("hello world");
    let mut parser = Parser::new(&source);

    while let Some(c) = parser.peek() {
        print!("[{}]", c);
        parser.advance();
    }
    println!();
    // [h][e][l][l][o][ ][w][o][r][l][d]
}
```

### 9.3 Anti-Pattern: Self-Referential Structs

```rust
// This CANNOT work in safe Rust:
// struct SelfRef {
//     data: String,
//     slice: &str, // wants to borrow from data above
// }
//
// Why? If SelfRef moves in memory, slice becomes a dangling pointer.
// The borrow checker prevents this at compile time.

// Solutions:
// 1. Store an index instead of a reference
struct TextWithHighlight {
    text: String,
    highlight_start: usize,
    highlight_end: usize,
}

impl TextWithHighlight {
    fn highlighted(&self) -> &str {
        &self.text[self.highlight_start..self.highlight_end]
    }
}

// 2. Use separate structs with explicit lifetime relationship
// 3. Use Pin + unsafe (advanced — beyond this lesson)
// 4. Use crates like ouroboros or self_cell
```

### 9.4 Anti-Pattern: Unnecessary Lifetime Annotations

```rust
// Unnecessary: the compiler applies elision Rule 2 automatically
// fn first_char<'a>(s: &'a str) -> &'a str {
//     &s[..1]
// }

// Better: let elision do its job
fn first_char(s: &str) -> &str {
    &s[..1]
}

// Only add lifetime annotations when the compiler asks for them
// or when you need to express a relationship between multiple references
```

### 9.5 Decision Flowchart

```
Do I need lifetime annotations?

  Does my function take references AND return a reference?
  ├── No  → You don't need them
  └── Yes → Does it take exactly ONE reference input?
            ├── Yes → Elision handles it (Rule 2)
            └── No  → Is one of the inputs &self?
                      ├── Yes → Elision handles it (Rule 3)
                      └── No  → You need explicit lifetimes!

  Does my struct hold references?
  ├── No  → You don't need them
  └── Yes → You MUST annotate the struct with lifetimes
```

---

## 10. Practice Problems

### Problem 1: First and Last
Write a function `first_and_last<'a>(words: &'a [String]) -> (&'a str, &'a str)` that returns references to the first and last strings in a slice. It should return `("", "")` for an empty slice. Verify that the returned references are valid by using them after the function call.

### Problem 2: Longest Line
Write a function `longest_line<'a>(text: &'a str) -> &'a str` that takes a multi-line string and returns a reference to the longest line. Use `.lines()` to iterate. Test it with a string that has lines of varying lengths.

### Problem 3: Struct with Lifetime
Create a `WordIterator<'a>` struct that holds a `&'a str` and iterates over words one at a time (like a manual implementation of `split_whitespace`). Implement `Iterator` for it with `Item = &'a str`. Test it with a `for` loop and with iterator methods like `.count()` and `.collect::<Vec<_>>()`.

### Problem 4: Multiple Lifetimes
Write a `Merger<'a, 'b>` struct that holds two string slices with **different** lifetimes. Implement a method `merge(&self) -> String` that concatenates them (returns owned data — no lifetime issues). Then implement `first(&self) -> &'a str` and `second(&self) -> &'b str` that return the individual references with their correct lifetimes. Demonstrate that the two source strings can have different scopes.

### Problem 5: Lifetime Debugging
The following code does not compile. Identify the lifetime issue, explain why it fails, and fix it (there may be multiple valid fixes):

```rust
fn longest_with_announcement(x: &str, y: &str, ann: &str) -> &str {
    println!("Announcement: {}", ann);
    if x.len() > y.len() { x } else { y }
}
```

Then write a version where `ann` has a **different** lifetime from `x` and `y`, demonstrating that the announcement does not need to live as long as the return value.

---

## References

- [The Rust Programming Language, Ch. 10.3: Validating References with Lifetimes](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html)
- [Rust by Example: Lifetimes](https://doc.rust-lang.org/rust-by-example/scope/lifetime.html)
- [The Rustonomicon: Lifetimes](https://doc.rust-lang.org/nomicon/lifetimes.html)
- [Common Rust Lifetime Misconceptions (pretzelhammer)](https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md)
- [Lifetime Elision Rules (Rust Reference)](https://doc.rust-lang.org/reference/lifetime-elision.html)

---

**Previous**: [Traits and Generics](./10_Traits_and_Generics.md) | **Next**: [Closures and Iterators](./12_Closures_and_Iterators.md)
