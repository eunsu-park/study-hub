# 07. Enums and Pattern Matching

**Previous**: [Structs and Methods](./06_Structs_and_Methods.md) | **Next**: [Collections](./08_Collections.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define enums with unit, tuple, and struct variants
2. Use `Option<T>` to handle the absence of values safely
3. Write exhaustive `match` expressions for control flow
4. Apply `if let` and `let else` for concise single-pattern matching

---

Enums and pattern matching are the backbone of Rust's type-driven design. Where other languages use null pointers, exception hierarchies, or type tags, Rust uses enums to encode every possible state into the type system — and `match` to handle each case exhaustively.

## Table of Contents
1. [Defining Enums](#1-defining-enums)
2. [Option Type](#2-option-type)
3. [Pattern Matching with match](#3-pattern-matching-with-match)
4. [if let and let else](#4-if-let-and-let-else)
5. [Real-World Enum Patterns](#5-real-world-enum-patterns)
6. [Practice Problems](#6-practice-problems)

---

## 1. Defining Enums

### 1.1 Basic Enums

```rust
#[derive(Debug)]
enum Direction {
    North,
    South,
    East,
    West,
}

fn describe(dir: &Direction) -> &str {
    match dir {
        Direction::North => "heading north",
        Direction::South => "heading south",
        Direction::East => "heading east",
        Direction::West => "heading west",
    }
}

fn main() {
    let dir = Direction::North;
    println!("{}: {}", dir, describe(&dir));
}
```

### 1.2 Enums with Data

Each variant can carry different types and amounts of data:

```rust
#[derive(Debug)]
enum Message {
    Quit,                       // Unit variant (no data)
    Move { x: i32, y: i32 },   // Struct variant (named fields)
    Write(String),              // Tuple variant (one field)
    ChangeColor(u8, u8, u8),   // Tuple variant (multiple fields)
}

impl Message {
    fn process(&self) {
        match self {
            Message::Quit => println!("Quitting"),
            Message::Move { x, y } => println!("Moving to ({x}, {y})"),
            Message::Write(text) => println!("Writing: {text}"),
            Message::ChangeColor(r, g, b) => println!("Color: ({r}, {g}, {b})"),
        }
    }
}

fn main() {
    let messages = vec![
        Message::Quit,
        Message::Move { x: 10, y: 20 },
        Message::Write(String::from("hello")),
        Message::ChangeColor(255, 0, 128),
    ];

    for msg in &messages {
        msg.process();
    }
}
```

### 1.3 Enums vs Structs

```
Struct: one shape, many instances
  struct Point { x: f64, y: f64 }  → Every Point has x and y

Enum: many shapes (variants), one type
  enum Shape { Circle(f64), Rect(f64, f64) }  → A Shape is EITHER Circle OR Rect
```

---

## 2. Option Type

Rust has no `null`. Instead, the `Option<T>` enum encodes the possibility of absence:

```rust
// Defined in the standard library:
// enum Option<T> {
//     Some(T),   // A value is present
//     None,      // No value
// }

fn divide(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 {
        None
    } else {
        Some(a / b)
    }
}

fn find_first_even(numbers: &[i32]) -> Option<i32> {
    for &n in numbers {
        if n % 2 == 0 {
            return Some(n);
        }
    }
    None
}

fn main() {
    // Using match
    match divide(10.0, 3.0) {
        Some(result) => println!("10 / 3 = {result:.2}"),
        None => println!("Cannot divide by zero"),
    }

    // Useful Option methods
    let x: Option<i32> = Some(42);
    let y: Option<i32> = None;

    println!("{}", x.unwrap());              // 42 (panics if None!)
    println!("{}", y.unwrap_or(0));           // 0 (safe default)
    println!("{}", y.unwrap_or_default());    // 0 (uses Default trait)
    println!("{}", x.is_some());             // true
    println!("{}", y.is_none());             // true

    // map — transform the inner value
    let doubled = x.map(|v| v * 2);  // Some(84)

    // and_then — chain operations that return Option (flatmap)
    let result = x
        .map(|v| v as f64)
        .and_then(|v| divide(v, 7.0));  // Some(6.0)
}
```

---

## 3. Pattern Matching with match

### 3.1 Exhaustive Matching

```rust
fn describe_number(n: i32) -> &'static str {
    match n {
        0 => "zero",
        1..=9 => "single digit",
        10..=99 => "double digit",
        100..=999 => "triple digit",
        _ => "large number",  // _ is the wildcard, catches everything else
    }
}
```

### 3.2 Destructuring in match

```rust
#[derive(Debug)]
enum Shape {
    Circle(f64),
    Rectangle(f64, f64),
    Triangle { base: f64, height: f64 },
}

fn area(shape: &Shape) -> f64 {
    match shape {
        Shape::Circle(radius) => std::f64::consts::PI * radius * radius,
        Shape::Rectangle(w, h) => w * h,
        Shape::Triangle { base, height } => 0.5 * base * height,
    }
}
```

### 3.3 Match Guards

```rust
fn classify(n: i32) -> &'static str {
    match n {
        n if n < 0 => "negative",
        0 => "zero",
        n if n % 2 == 0 => "positive even",
        _ => "positive odd",
    }
}
```

### 3.4 Binding with @

```rust
fn check_age(age: u32) {
    match age {
        0 => println!("newborn"),
        a @ 1..=12 => println!("child, age {a}"),
        a @ 13..=17 => println!("teenager, age {a}"),
        a => println!("adult, age {a}"),
    }
}
```

### 3.5 match is an Expression

```rust
fn main() {
    let n = 42;
    let description = match n % 3 {
        0 => "divisible by 3",
        1 => "remainder 1",
        2 => "remainder 2",
        _ => unreachable!(),
    };
    println!("{n} is {description}");
}
```

---

## 4. if let and let else

### 4.1 if let — Single Pattern

When you only care about one variant:

```rust
fn main() {
    let config_max: Option<u32> = Some(3);

    // Verbose match
    match config_max {
        Some(max) => println!("Max: {max}"),
        None => {},
    }

    // Concise if let
    if let Some(max) = config_max {
        println!("Max: {max}");
    }

    // if let with else
    if let Some(max) = config_max {
        println!("Max: {max}");
    } else {
        println!("No max configured");
    }
}
```

### 4.2 let else — Early Return Pattern

```rust
fn parse_config(input: &str) -> Option<u32> {
    // let else: bind or diverge (return, break, continue, panic)
    let Some(value) = input.strip_prefix("port=") else {
        return None;
    };

    let Ok(port) = value.parse::<u32>() else {
        return None;
    };

    Some(port)
}

fn main() {
    println!("{:?}", parse_config("port=8080"));  // Some(8080)
    println!("{:?}", parse_config("host=local"));  // None
}
```

### 4.3 while let

```rust
fn main() {
    let mut stack = vec![1, 2, 3];

    // Pop elements until the stack is empty
    while let Some(top) = stack.pop() {
        println!("Popped: {top}");
    }
}
```

---

## 5. Real-World Enum Patterns

### 5.1 State Machine

```rust
#[derive(Debug)]
enum ConnectionState {
    Disconnected,
    Connecting { attempt: u32 },
    Connected { session_id: String },
    Error(String),
}

impl ConnectionState {
    fn next(self) -> Self {
        match self {
            Self::Disconnected => Self::Connecting { attempt: 1 },
            Self::Connecting { attempt } if attempt < 3 => {
                Self::Connecting { attempt: attempt + 1 }
            }
            Self::Connecting { attempt } if attempt >= 3 => {
                Self::Error(format!("Failed after {attempt} attempts"))
            }
            _ => self,
        }
    }
}
```

### 5.2 Command Pattern

```rust
enum Command {
    Add(String),
    Remove(usize),
    List,
    Quit,
}

fn execute(cmd: &Command, items: &mut Vec<String>) {
    match cmd {
        Command::Add(item) => {
            items.push(item.clone());
            println!("Added: {item}");
        }
        Command::Remove(index) => {
            if *index < items.len() {
                let removed = items.remove(*index);
                println!("Removed: {removed}");
            }
        }
        Command::List => {
            for (i, item) in items.iter().enumerate() {
                println!("{i}: {item}");
            }
        }
        Command::Quit => println!("Goodbye!"),
    }
}
```

---

## 6. Practice Problems

### Exercise 1: Traffic Light
Define a `TrafficLight` enum with `Red`, `Yellow`, `Green` variants. Implement a `duration(&self) -> u32` method returning the seconds for each light and a `next(&self) -> Self` method.

### Exercise 2: Expression Evaluator
Define an `Expr` enum with `Num(f64)`, `Add(Box<Expr>, Box<Expr>)`, `Mul(Box<Expr>, Box<Expr>)`. Implement `fn eval(expr: &Expr) -> f64` using match.

### Exercise 3: Safe Division
Write `fn safe_div(a: i32, b: i32) -> Option<i32>` that returns `None` for division by zero. Chain three divisions using `.and_then()`.

### Exercise 4: Parse Command
Write a function that parses strings like "add milk", "remove 2", "list", "quit" into a `Command` enum. Return `Option<Command>`.

### Exercise 5: Nested Patterns
Write a function that takes `Option<Option<i32>>` and uses nested pattern matching to return the inner value, 0 for `Some(None)`, and -1 for `None`.

---

## References
- [The Rust Book — Enums](https://doc.rust-lang.org/book/ch06-00-enums.html)
- [The Rust Book — match](https://doc.rust-lang.org/book/ch06-02-match.html)
- [Rust by Example — enum](https://doc.rust-lang.org/rust-by-example/custom_types/enum.html)

---

**Previous**: [Structs and Methods](./06_Structs_and_Methods.md) | **Next**: [Collections](./08_Collections.md)
