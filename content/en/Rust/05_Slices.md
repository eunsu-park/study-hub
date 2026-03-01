# 05. Slices

**Previous**: [Borrowing and References](./04_Borrowing_and_References.md) | **Next**: [Structs and Methods](./06_Structs_and_Methods.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create string slices (`&str`) and array slices (`&[T]`) from owned data
2. Explain the difference between `String` and `&str` and when to use each
3. Write functions that accept `&str` to maximize flexibility
4. Use slice methods for searching, splitting, and iterating

---

Slices are references to a contiguous sequence of elements rather than the whole collection. They let you work with a portion of data without copying it, and they are the idiomatic way to pass strings and arrays in Rust.

## Table of Contents
1. [String Slices](#1-string-slices)
2. [String vs &str](#2-string-vs-str)
3. [Array and Vec Slices](#3-array-and-vec-slices)
4. [Slice Methods](#4-slice-methods)
5. [Practice Problems](#5-practice-problems)

---

## 1. String Slices

A **string slice** (`&str`) is a reference to a portion of a `String` (or string literal):

```rust
fn main() {
    let s = String::from("hello world");

    let hello = &s[0..5];   // "hello"
    let world = &s[6..11];  // "world"

    // Shorthand
    let hello = &s[..5];    // From start
    let world = &s[6..];    // To end
    let full = &s[..];      // Entire string

    println!("{hello} {world}");
}
```

```
String::from("hello world")

Stack:                          Heap:
s: [ptr, len=11, cap=11] ────→ [h|e|l|l|o| |w|o|r|l|d]
                                 0 1 2 3 4 5 6 7 8 9 10

hello: &s[0..5]
[ptr, len=5] ─────────────────→ [h|e|l|l|o]
                                 ↑ points into same heap memory

world: &s[6..11]
[ptr, len=5] ─────────────────→ [w|o|r|l|d]
                                 ↑ offset 6 into same allocation
```

### 1.1 UTF-8 and Slice Boundaries

Rust strings are UTF-8 encoded. Slicing at byte boundaries that fall inside a multi-byte character panics:

```rust
fn main() {
    let emoji = String::from("🦀 Rust");
    // let slice = &emoji[0..2];  // PANIC: byte 2 is inside the 🦀 codepoint (4 bytes)
    let slice = &emoji[0..4];     // OK: "🦀" (complete codepoint)
    let rest = &emoji[5..];       // "Rust"

    // Safe alternatives for character-level operations:
    for ch in emoji.chars() {
        print!("{ch} ");  // 🦀   R u s t
    }
}
```

---

## 2. String vs &str

| Feature | `String` | `&str` |
|---------|----------|--------|
| Ownership | Owned | Borrowed |
| Mutability | Growable (`push`, `push_str`) | Immutable view |
| Storage | Heap-allocated | Points to heap, stack, or binary |
| Size | ptr + len + capacity (24 bytes) | ptr + len (16 bytes) |
| Use case | Building/modifying strings | Reading/passing strings |

```rust
// String literals are &str — they live in the compiled binary
let literal: &str = "hello";

// String is heap-allocated and owned
let owned: String = String::from("hello");
let also_owned: String = "hello".to_string();

// &str from a String (cheap — just a pointer)
let slice: &str = &owned;
let slice: &str = owned.as_str();

// String from &str (allocates)
let new_string: String = literal.to_string();
let new_string: String = String::from(literal);
```

### 2.1 Idiomatic Function Signatures

```rust
// GOOD: accepts both String and &str
fn greet(name: &str) {
    println!("Hello, {name}!");
}

// Less flexible: only accepts String
fn greet_owned(name: String) {
    println!("Hello, {name}!");
}

fn main() {
    let owned = String::from("Alice");
    let literal = "Bob";

    greet(&owned);    // &String coerces to &str automatically (deref coercion)
    greet(literal);   // &str passed directly

    greet_owned(owned);    // Moves the String
    // greet_owned(literal);  // ERROR: expected String, found &str
}
```

> **Rule of thumb**: Accept `&str` in function parameters; return `String` when you need to give the caller ownership.

---

## 3. Array and Vec Slices

Slices work for arrays and `Vec<T>` too:

```rust
fn sum(numbers: &[i32]) -> i32 {
    numbers.iter().sum()
}

fn main() {
    // Slice from array
    let arr = [1, 2, 3, 4, 5];
    let slice = &arr[1..4];  // [2, 3, 4]
    println!("sum of slice: {}", sum(slice));

    // Slice from Vec
    let vec = vec![10, 20, 30, 40, 50];
    let slice = &vec[..3];   // [10, 20, 30]
    println!("sum of vec slice: {}", sum(slice));

    // Entire collection as slice
    println!("sum of all: {}", sum(&arr));   // &[i32; 5] coerces to &[i32]
    println!("sum of all: {}", sum(&vec));   // &Vec<i32> coerces to &[i32]
}
```

### 3.1 Mutable Slices

```rust
fn zero_out(data: &mut [i32]) {
    for element in data.iter_mut() {
        *element = 0;
    }
}

fn main() {
    let mut numbers = [1, 2, 3, 4, 5];
    zero_out(&mut numbers[1..4]);
    println!("{numbers:?}");  // [1, 0, 0, 0, 5]
}
```

---

## 4. Slice Methods

### 4.1 String Slice Methods

```rust
fn main() {
    let s = "Hello, Rust World!";

    // Searching
    println!("{}", s.contains("Rust"));      // true
    println!("{}", s.starts_with("Hello"));  // true
    println!("{:?}", s.find("Rust"));        // Some(7)

    // Splitting
    let words: Vec<&str> = s.split_whitespace().collect();
    println!("{words:?}");  // ["Hello,", "Rust", "World!"]

    let parts: Vec<&str> = "a,b,c".split(',').collect();
    println!("{parts:?}");  // ["a", "b", "c"]

    // Trimming
    let padded = "  hello  ";
    println!("'{}'", padded.trim());        // 'hello'
    println!("'{}'", padded.trim_start());  // 'hello  '

    // Replacing
    let replaced = s.replace("Rust", "Ferris");
    println!("{replaced}");  // Hello, Ferris World!

    // Case conversion (returns new String)
    println!("{}", s.to_uppercase());
    println!("{}", s.to_lowercase());
}
```

### 4.2 Array/Vec Slice Methods

```rust
fn main() {
    let data = [3, 1, 4, 1, 5, 9, 2, 6];

    // Searching
    println!("{}", data.contains(&5));           // true
    println!("{:?}", data.iter().position(|&x| x == 9)); // Some(5)

    // Windowing
    for window in data.windows(3) {
        print!("{window:?} ");  // [3,1,4] [1,4,1] [4,1,5] ...
    }
    println!();

    // Chunking
    for chunk in data.chunks(3) {
        print!("{chunk:?} ");  // [3,1,4] [1,5,9] [2,6]
    }
    println!();

    // Sorting (requires mutable slice)
    let mut sorted = data;
    sorted.sort();
    println!("{sorted:?}");  // [1, 1, 2, 3, 4, 5, 6, 9]

    // Binary search (on sorted data)
    println!("{:?}", sorted.binary_search(&4));  // Ok(4)
}
```

---

## 5. Practice Problems

### Exercise 1: First Word
Write `fn first_word(s: &str) -> &str` that returns the first word of a string (text before the first space, or the entire string if no space).

### Exercise 2: String Reversal
Write `fn reverse_words(s: &str) -> String` that reverses the order of words. `"hello world"` → `"world hello"`.

### Exercise 3: Slice Sum
Write `fn moving_average(data: &[f64], window: usize) -> Vec<f64>` that computes a moving average using `data.windows(window)`.

### Exercise 4: &str vs String
Refactor a function from `fn process(s: String)` to `fn process(s: &str)`. Explain why the refactored version is more flexible.

### Exercise 5: Safe Substring
Write `fn safe_substring(s: &str, start: usize, end: usize) -> Option<&str>` that returns `None` if the range is invalid or falls on non-UTF-8 boundaries, instead of panicking.

---

## References
- [The Rust Book — The Slice Type](https://doc.rust-lang.org/book/ch04-03-slices.html)
- [std::str documentation](https://doc.rust-lang.org/std/primitive.str.html)
- [std::slice documentation](https://doc.rust-lang.org/std/primitive.slice.html)

---

**Previous**: [Borrowing and References](./04_Borrowing_and_References.md) | **Next**: [Structs and Methods](./06_Structs_and_Methods.md)
