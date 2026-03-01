# 02. Variables and Types

**Previous**: [Getting Started](./01_Getting_Started.md) | **Next**: [Ownership](./03_Ownership.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare immutable and mutable variables using `let` and `let mut`
2. Use shadowing to rebind variables with a new type or value
3. Differentiate between scalar types (integers, floats, bool, char) and compound types (tuples, arrays)
4. Apply type inference and explicit type annotations appropriately
5. Explain why Rust defaults to immutability and how `const` differs from `let`

---

Rust's type system is one of its greatest strengths. Every value has a known type at compile time, yet you rarely need to write types explicitly — the compiler infers them. Combined with immutability by default, this creates code that is both concise and resistant to accidental mutation bugs.

## Table of Contents
1. [Variables and Mutability](#1-variables-and-mutability)
2. [Scalar Types](#2-scalar-types)
3. [Compound Types](#3-compound-types)
4. [Type Conversion](#4-type-conversion)
5. [Constants and Statics](#5-constants-and-statics)
6. [Practice Problems](#6-practice-problems)

---

## 1. Variables and Mutability

### 1.1 Immutable by Default

```rust
fn main() {
    let x = 5;
    // x = 6;  // ERROR: cannot assign twice to immutable variable
    println!("x = {x}");
}
```

Why default to immutable? When a variable never changes, you can reason about the code locally — no need to trace all possible mutations across the program. The compiler enforces this guarantee.

### 1.2 Mutable Variables

```rust
fn main() {
    let mut counter = 0;  // mut keyword opts into mutability
    counter += 1;
    counter += 1;
    println!("counter = {counter}");  // 2
}
```

### 1.3 Shadowing

Shadowing redeclares a variable with the same name. Unlike mutation, shadowing can change the type:

```rust
fn main() {
    let x = 5;          // x is i32
    let x = x + 1;      // New x shadows the old one (still i32)
    let x = x * 2;      // Shadows again
    println!("x = {x}"); // 12

    // Shadowing can change the type
    let spaces = "   ";         // &str
    let spaces = spaces.len();  // usize — different type, same name
    println!("spaces = {spaces}"); // 3

    // With mut, you CANNOT change the type:
    // let mut s = "hello";
    // s = s.len();  // ERROR: expected &str, found usize
}
```

Shadowing is idiomatic in Rust for transforming a value through a pipeline of operations while keeping a meaningful name.

### 1.4 Type Annotations

```rust
fn main() {
    // Explicit type annotation
    let x: i32 = 42;
    let pi: f64 = 3.14159;
    let active: bool = true;

    // Type inference — compiler deduces the type
    let y = 42;        // inferred as i32 (default integer type)
    let z = 3.14;      // inferred as f64 (default float type)
    let name = "Rust";  // inferred as &str
}
```

---

## 2. Scalar Types

Scalar types represent a single value.

### 2.1 Integer Types

| Size | Signed | Unsigned |
|------|--------|----------|
| 8-bit | `i8` | `u8` |
| 16-bit | `i16` | `u16` |
| 32-bit | `i32` (default) | `u32` |
| 64-bit | `i64` | `u64` |
| 128-bit | `i128` | `u128` |
| Pointer-sized | `isize` | `usize` |

```rust
fn main() {
    let decimal = 98_222;      // Underscores for readability
    let hex = 0xff;            // Hexadecimal
    let octal = 0o77;          // Octal
    let binary = 0b1111_0000;  // Binary
    let byte = b'A';           // Byte literal (u8 only)

    // Integer overflow behavior:
    // - Debug mode:   panics at runtime
    // - Release mode: wraps around (two's complement)
    // Use wrapping_*, checked_*, overflowing_*, saturating_* for explicit control
    let max: u8 = 255;
    let wrapped = max.wrapping_add(1);  // 0
    let saturated = max.saturating_add(1);  // 255
    let checked = max.checked_add(1);  // None
}
```

### 2.2 Floating-Point Types

```rust
fn main() {
    let x = 2.0;      // f64 (default, double precision)
    let y: f32 = 3.0;  // f32 (single precision)

    // Arithmetic
    let sum = 5.0 + 10.0;
    let difference = 95.5 - 4.3;
    let product = 4.0 * 30.0;
    let quotient = 56.7 / 32.2;
    let remainder = 43.0 % 5.0;  // 3.0

    // f64 is generally preferred — same speed as f32 on modern CPUs
    // but more precision (15-17 significant digits vs 6-9)
}
```

### 2.3 Boolean and Character

```rust
fn main() {
    // Boolean — 1 byte
    let t: bool = true;
    let f = false;

    // Character — 4 bytes (Unicode scalar value)
    let c = 'z';
    let emoji = '🦀';
    let hangul = '가';

    // char represents a Unicode scalar value (U+0000 to U+D7FF, U+E000 to U+10FFFF)
    // This is NOT the same as a byte — a char is always 4 bytes in Rust
    println!("size of char: {} bytes", std::mem::size_of::<char>());  // 4
}
```

---

## 3. Compound Types

### 3.1 Tuples

Tuples group values of different types into one compound value. They have a fixed length.

```rust
fn main() {
    // Creating a tuple
    let tup: (i32, f64, u8) = (500, 6.4, 1);

    // Destructuring
    let (x, y, z) = tup;
    println!("y = {y}");  // 6.4

    // Index access (zero-based)
    let five_hundred = tup.0;
    let six_point_four = tup.1;
    let one = tup.2;

    // Unit tuple — the empty tuple () is Rust's "void"
    let unit: () = ();
    // Functions without a return value implicitly return ()
}
```

### 3.2 Arrays

Arrays have a fixed length and store elements of the same type. They live on the **stack**.

```rust
fn main() {
    // Array declaration
    let a: [i32; 5] = [1, 2, 3, 4, 5];

    // Initialize with same value
    let zeros = [0; 10];  // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    // Access elements
    let first = a[0];
    let second = a[1];

    // Rust checks array bounds at runtime
    // let invalid = a[10];  // Panics: index out of bounds

    // Array length
    println!("length: {}", a.len());  // 5

    // Iteration
    for element in &a {
        println!("{element}");
    }

    // Arrays are useful when you know the exact size at compile time
    // For dynamic sizes, use Vec<T> (covered in Lesson 08)
}
```

### 3.3 Strings (Preview)

Rust has two main string types. This is a brief introduction — Lesson 05 (Slices) covers them in depth.

```rust
fn main() {
    // &str — string slice, immutable reference to string data
    let greeting: &str = "Hello, world!";  // Stored in binary

    // String — heap-allocated, growable, owned string
    let mut name = String::from("Rust");
    name.push_str(" Programming");
    println!("{name}");  // Rust Programming

    // Converting between them
    let s: String = greeting.to_string();
    let slice: &str = &s;
}
```

---

## 4. Type Conversion

Rust has no implicit type conversions (coercions) between numeric types. You must be explicit:

```rust
fn main() {
    // `as` keyword for primitive casts
    let x: i32 = 42;
    let y: f64 = x as f64;
    let z: u8 = x as u8;  // May truncate!

    // Safer conversions with From/Into traits
    let a: i32 = 5;
    let b: i64 = i64::from(a);  // Infallible widening conversion
    let c: i64 = a.into();      // Same thing, using Into trait

    // TryFrom for fallible conversions
    let big: i64 = 1_000_000;
    let small: Result<i32, _> = i32::try_from(big);  // Ok(1000000)

    let too_big: i64 = 5_000_000_000;
    let fail: Result<i32, _> = i32::try_from(too_big);  // Err(...)

    // String to number
    let parsed: i32 = "42".parse().expect("not a number");
    let pi: f64 = "3.14".parse().unwrap();
}
```

---

## 5. Constants and Statics

### 5.1 Constants

```rust
// Constants must have a type annotation and be known at compile time
const MAX_POINTS: u32 = 100_000;
const PI: f64 = 3.141_592_653_589_793;

fn main() {
    // Constants are inlined at each usage site
    println!("Max: {MAX_POINTS}");
}
```

### 5.2 Static Variables

```rust
// Static variables have a fixed memory address for the entire program
static LANGUAGE: &str = "Rust";
static mut COUNTER: u32 = 0;  // Mutable statics require unsafe to access

fn main() {
    println!("{LANGUAGE}");

    // Mutable statics are inherently unsafe (data races possible)
    unsafe {
        COUNTER += 1;
        println!("COUNTER = {COUNTER}");
    }
}
```

| Feature | `const` | `static` | `let` |
|---------|---------|----------|-------|
| Scope | Any | Any | Block |
| Memory | Inlined | Fixed address | Stack |
| Mutability | Never | `static mut` (unsafe) | `let mut` |
| Type annotation | Required | Required | Optional |
| Computed at | Compile time | Compile time | Runtime |

---

## 6. Practice Problems

### Exercise 1: Variable Binding
Declare a variable `temperature` with value `36.6`, then shadow it with the integer part only (use `as` to cast). Print both the original concept and the shadowed value.

### Exercise 2: Tuple Destructuring
Create a function `min_max(a: i32, b: i32, c: i32) -> (i32, i32)` that returns the minimum and maximum of three numbers. Destructure the result in `main`.

### Exercise 3: Array Operations
Create an array of 12 monthly rainfall values (made-up f64 values). Write a loop that computes the total annual rainfall and the average monthly rainfall.

### Exercise 4: Type Conversions
Write code that converts the string `"255"` to `u8` using `.parse()`. Then try parsing `"256"` and handle the error gracefully using `match`. What error type does `.parse::<u8>()` return on failure?

### Exercise 5: Overflow Behavior
Demonstrate the difference between `wrapping_add`, `checked_add`, and `saturating_add` on `u8::MAX`. Which method would you use for a network packet sequence counter? Why?

---

## References
- [The Rust Book — Variables and Mutability](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html)
- [The Rust Book — Data Types](https://doc.rust-lang.org/book/ch03-02-data-types.html)
- [Rust Reference — Types](https://doc.rust-lang.org/reference/types.html)

---

**Previous**: [Getting Started](./01_Getting_Started.md) | **Next**: [Ownership](./03_Ownership.md)
