# 10. Traits and Generics

**Previous**: [Error Handling](./09_Error_Handling.md) | **Next**: [Lifetimes](./11_Lifetimes.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define traits with methods and default implementations, and implement them for custom types
2. Write generic functions and structs with trait bounds using both inline syntax and `where` clauses
3. Distinguish between static dispatch (generics) and dynamic dispatch (`dyn Trait`) and choose appropriately
4. Implement key standard library traits (`Display`, `From`, `Iterator`, `Default`) for your own types
5. Use supertraits to express trait hierarchies

---

Traits and generics are the twin pillars of Rust's approach to abstraction. Traits define shared behavior — like interfaces in Java or protocols in Swift — while generics let you write code that works across many types without sacrificing performance. Together, they give you the flexibility of dynamically typed languages with the speed of statically typed ones.

An analogy: think of a trait as a job description ("must be able to sort, compare, and display"), and generics as a hiring policy that says "anyone who meets this job description can fill this role." The compiler then generates specialized code for each specific type that fills the role — no runtime overhead.

## Table of Contents
1. [Defining Traits](#1-defining-traits)
2. [Implementing Traits](#2-implementing-traits)
3. [Trait Bounds](#3-trait-bounds)
4. [impl Trait Syntax](#4-impl-trait-syntax)
5. [where Clauses](#5-where-clauses)
6. [Generic Structs and Enums](#6-generic-structs-and-enums)
7. [Standard Library Traits](#7-standard-library-traits)
8. [Trait Objects: Dynamic Dispatch](#8-trait-objects-dynamic-dispatch)
9. [Supertraits and Trait Inheritance](#9-supertraits-and-trait-inheritance)
10. [Practice Problems](#10-practice-problems)

---

## 1. Defining Traits

A trait defines a set of methods that a type can implement. Some methods can have default implementations:

```rust
trait Summary {
    // Required method — implementors MUST define this
    fn summarize_author(&self) -> String;

    // Default method — implementors CAN override this
    fn summarize(&self) -> String {
        // Default implementation calls the required method
        format!("(Read more from {}...)", self.summarize_author())
    }
}
```

### 1.1 Traits with Multiple Methods

```rust
trait Shape {
    fn area(&self) -> f64;
    fn perimeter(&self) -> f64;

    // Default: uses other methods defined in the same trait
    fn describe(&self) -> String {
        format!(
            "Shape with area {:.2} and perimeter {:.2}",
            self.area(),
            self.perimeter()
        )
    }
}
```

### 1.2 Traits as Contracts

```
Trait: Summary
┌──────────────────────────────┐
│  Required:                   │
│    fn summarize_author(&self)│
│                              │
│  Provided (default):         │
│    fn summarize(&self)       │
│      → calls summarize_author│
└──────────────────────────────┘
        │
        │ impl Summary for ...
        ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  NewsArticle  │  │   Tweet      │  │   BlogPost   │
│  ✓ author()   │  │  ✓ author()  │  │  ✓ author()  │
│  ✓ summarize()│  │  (default)   │  │  ✓ summarize│
│    (custom)   │  │              │  │    (custom)  │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 2. Implementing Traits

Use `impl TraitName for TypeName` to implement a trait:

```rust
trait Summary {
    fn summarize_author(&self) -> String;
    fn summarize(&self) -> String {
        format!("(Read more from {}...)", self.summarize_author())
    }
}

struct NewsArticle {
    title: String,
    author: String,
    content: String,
}

impl Summary for NewsArticle {
    fn summarize_author(&self) -> String {
        self.author.clone()
    }

    // Override the default summarize()
    fn summarize(&self) -> String {
        format!("{}, by {} — {}", self.title, self.author, &self.content[..50.min(self.content.len())])
    }
}

struct Tweet {
    username: String,
    text: String,
}

impl Summary for Tweet {
    fn summarize_author(&self) -> String {
        format!("@{}", self.username)
    }
    // Uses the default summarize() — no need to override
}

fn main() {
    let article = NewsArticle {
        title: String::from("Rust 2025 Edition Released"),
        author: String::from("Niko Matsakis"),
        content: String::from("The Rust team announced the 2025 edition today with exciting new features..."),
    };

    let tweet = Tweet {
        username: String::from("rustlang"),
        text: String::from("Rust 2025 is here!"),
    };

    println!("{}", article.summarize());
    // "Rust 2025 Edition Released, by Niko Matsakis — The Rust team announced..."

    println!("{}", tweet.summarize());
    // "(Read more from @rustlang...)"
}
```

### 2.1 The Orphan Rule

You can only implement a trait for a type if **at least one** of them is local to your crate. This prevents conflicting implementations:

```rust
// In YOUR crate:

// OK: your trait on a foreign type
// impl MyTrait for Vec<i32> { ... }

// OK: a foreign trait on your type
// impl Display for MyStruct { ... }

// NOT OK: a foreign trait on a foreign type
// impl Display for Vec<i32> { ... }  // compiler error!
```

---

## 3. Trait Bounds

Trait bounds constrain generic types. They say "T can be any type, as long as it implements this trait":

```rust
use std::fmt::Display;

// T must implement both Display and PartialOrd
fn largest<T: Display + PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in &list[1..] {
        if item > largest {
            largest = item;
        }
    }
    println!("The largest is {}", largest); // Display lets us print
    largest
}

fn main() {
    let numbers = vec![34, 50, 25, 100, 65];
    largest(&numbers); // T = i32, which is Display + PartialOrd

    let chars = vec!['y', 'm', 'a', 'q'];
    largest(&chars); // T = char, also Display + PartialOrd
}
```

### 3.1 Multiple Trait Bounds

Use `+` to require multiple traits:

```rust
use std::fmt::{Display, Debug};

fn print_both<T: Display + Debug>(item: &T) {
    println!("Display: {}", item);    // uses Display
    println!("Debug:   {:?}", item);  // uses Debug
}

fn main() {
    print_both(&42);
    print_both(&"hello");
}
```

---

## 4. impl Trait Syntax

`impl Trait` is syntactic sugar that works in two positions:

### 4.1 Argument Position (Sugar for Trait Bounds)

```rust
use std::fmt::Display;

// These two signatures are equivalent:
fn notify_verbose<T: Display>(item: &T) {
    println!("Breaking: {}", item);
}

fn notify(item: &impl Display) {
    println!("Breaking: {}", item);
}
// "impl Display" in argument position means "some type that implements Display"

fn main() {
    notify(&"earthquake");
    notify(&42);
}
```

### 4.2 Return Position (Opaque Return Types)

```rust
fn make_greeting(name: &str) -> impl std::fmt::Display {
    // The caller knows the return type implements Display,
    // but NOT which concrete type it is (it's String here)
    format!("Hello, {}!", name)
}

fn main() {
    let greeting = make_greeting("Rust");
    println!("{}", greeting); // works because it implements Display
    // But you can't call String-specific methods on greeting
}
```

Return-position `impl Trait` is especially useful with closures and iterators, which have unnameable types:

```rust
fn make_adder(x: i32) -> impl Fn(i32) -> i32 {
    // Each closure has a unique, anonymous type — impl Fn lets us return it
    move |y| x + y
}

fn even_numbers(limit: i32) -> impl Iterator<Item = i32> {
    (0..limit).filter(|n| n % 2 == 0)
}

fn main() {
    let add_five = make_adder(5);
    println!("{}", add_five(3)); // 8

    for n in even_numbers(10) {
        print!("{} ", n); // 0 2 4 6 8
    }
    println!();
}
```

---

## 5. where Clauses

When trait bounds get complex, `where` clauses improve readability:

```rust
use std::fmt::{Debug, Display};
use std::hash::Hash;

// Cluttered: hard to read with inline bounds
fn process_cluttered<T: Display + Debug + Clone + Hash, U: Debug + Default>(t: &T, u: &U) {
    println!("{:?} {:?}", t, u);
}

// Clean: where clause separates bounds from the signature
fn process<T, U>(t: &T, u: &U)
where
    T: Display + Debug + Clone + Hash,
    U: Debug + Default,
{
    println!("{:?} {:?}", t, u);
}

fn main() {
    process(&42, &String::new());
}
```

`where` clauses can also express bounds that inline syntax cannot:

```rust
use std::fmt::Debug;

// Bound on an associated type — only possible with where
fn print_pairs<T>(items: &T)
where
    T: IntoIterator + Clone,
    T::Item: Debug, // bound on the associated type Item
{
    for item in items.clone() {
        println!("{:?}", item);
    }
}

fn main() {
    print_pairs(&vec![1, 2, 3]);
}
```

---

## 6. Generic Structs and Enums

Generics are not limited to functions — structs and enums can be generic too:

### 6.1 Generic Structs

```rust
#[derive(Debug)]
struct Point<T> {
    x: T,
    y: T,
}

// Implement methods for ALL Point<T>
impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Point { x, y }
    }
}

// Implement methods ONLY for Point<f64>
impl Point<f64> {
    fn distance_from_origin(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

fn main() {
    let int_point = Point::new(5, 10);
    let float_point = Point::new(1.0, 5.0);

    println!("{:?}", int_point);
    println!("distance: {:.2}", float_point.distance_from_origin());
    // int_point.distance_from_origin(); // ERROR: only defined for f64
}
```

### 6.2 Multiple Type Parameters

```rust
#[derive(Debug)]
struct Pair<A, B> {
    first: A,
    second: B,
}

impl<A, B> Pair<A, B> {
    fn new(first: A, second: B) -> Self {
        Pair { first, second }
    }

    // Method that mixes type parameters from two different Pairs
    fn mix<C, D>(self, other: Pair<C, D>) -> Pair<A, D> {
        Pair {
            first: self.first,
            second: other.second,
        }
    }
}

fn main() {
    let p1 = Pair::new("hello", 42);
    let p2 = Pair::new(3.14, true);
    let mixed = p1.mix(p2);
    println!("{:?}", mixed); // Pair { first: "hello", second: true }
}
```

### 6.3 Generic Enums

You have already used the two most famous generic enums:

```rust
// From the standard library:
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Your own generic enum
#[derive(Debug)]
enum Tree<T> {
    Leaf(T),
    Node {
        value: T,
        left: Box<Tree<T>>,
        right: Box<Tree<T>>,
    },
}

fn main() {
    let tree = Tree::Node {
        value: 10,
        left: Box::new(Tree::Leaf(5)),
        right: Box::new(Tree::Node {
            value: 15,
            left: Box::new(Tree::Leaf(12)),
            right: Box::new(Tree::Leaf(20)),
        }),
    };
    println!("{:#?}", tree);
}
```

---

## 7. Standard Library Traits

These traits appear everywhere in Rust. Implementing them for your types unlocks powerful functionality:

### 7.1 Display — Human-Readable Output

```rust
use std::fmt;

struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }
}

fn main() {
    let red = Color { r: 255, g: 0, b: 0 };
    println!("{}", red);   // #FF0000
    // println!("{:?}", red); // would need #[derive(Debug)]
}
```

### 7.2 From and Into — Type Conversions

```rust
struct Celsius(f64);
struct Fahrenheit(f64);

// Implementing From<X> for Y automatically gives you Into<Y> for X
impl From<Celsius> for Fahrenheit {
    fn from(c: Celsius) -> Self {
        Fahrenheit(c.0 * 9.0 / 5.0 + 32.0)
    }
}

impl From<Fahrenheit> for Celsius {
    fn from(f: Fahrenheit) -> Self {
        Celsius((f.0 - 32.0) * 5.0 / 9.0)
    }
}

fn main() {
    let boiling = Celsius(100.0);

    // Using From explicitly
    let f = Fahrenheit::from(boiling);
    println!("{}F", f.0); // 212F

    // Using Into (available automatically)
    let body_temp = Fahrenheit(98.6);
    let c: Celsius = body_temp.into();
    println!("{:.1}C", c.0); // 37.0C
}
```

### 7.3 Iterator — Making Your Type Iterable

```rust
struct Countdown {
    value: i32,
}

impl Countdown {
    fn new(start: i32) -> Self {
        Countdown { value: start }
    }
}

impl Iterator for Countdown {
    type Item = i32; // the associated type — what the iterator yields

    fn next(&mut self) -> Option<Self::Item> {
        if self.value > 0 {
            let current = self.value;
            self.value -= 1;
            Some(current)
        } else {
            None // signals the end of iteration
        }
    }
}

fn main() {
    // Now Countdown works with for loops and all iterator adaptors
    for n in Countdown::new(5) {
        print!("{} ", n);
    }
    println!(); // 5 4 3 2 1

    // And with iterator methods
    let sum: i32 = Countdown::new(10).filter(|n| n % 2 == 0).sum();
    println!("Sum of even numbers 1-10: {}", sum); // 2+4+6+8+10 = 30
}
```

### 7.4 Default — Sensible Defaults

```rust
#[derive(Debug)]
struct Config {
    host: String,
    port: u16,
    max_connections: usize,
    verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            host: String::from("localhost"),
            port: 8080,
            max_connections: 100,
            verbose: false,
        }
    }
}

fn main() {
    // Create with all defaults
    let config = Config::default();
    println!("{:?}", config);

    // Override specific fields using struct update syntax
    let custom = Config {
        port: 3000,
        verbose: true,
        ..Config::default() // fill the rest from default
    };
    println!("{:?}", custom);
}
```

---

## 8. Trait Objects: Dynamic Dispatch

Generics use **static dispatch** — the compiler generates a separate version of the function for each concrete type (monomorphization). This is fast but increases binary size.

**Trait objects** use **dynamic dispatch** — a single function handles all types through a vtable pointer at runtime:

```
Static Dispatch (generics):            Dynamic Dispatch (dyn Trait):

  fn draw<T: Shape>(s: &T)              fn draw(s: &dyn Shape)
           │                                      │
    ┌──────┼──────┐                    ┌──────────┴──────────┐
    ▼      ▼      ▼                    ▼                     │
draw_Circle  draw_Rect  draw_Tri     draw(...) single fn     │
(specialized  (specialized            │                      │
 for each      for each)              uses vtable            │
 type)                                at runtime             │
                                      ┌──────────────┐       │
                                      │ vtable       │       │
                                      │  area → ...  │       │
                                      │  draw → ...  │       │
                                      └──────────────┘       │
```

```rust
trait Shape {
    fn area(&self) -> f64;
    fn name(&self) -> &str;
}

struct Circle { radius: f64 }
struct Rectangle { width: f64, height: f64 }

impl Shape for Circle {
    fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius }
    fn name(&self) -> &str { "Circle" }
}

impl Shape for Rectangle {
    fn area(&self) -> f64 { self.width * self.height }
    fn name(&self) -> &str { "Rectangle" }
}

// Dynamic dispatch: accepts ANY type that implements Shape
fn print_area(shape: &dyn Shape) {
    println!("{}: area = {:.2}", shape.name(), shape.area());
}

fn main() {
    let circle = Circle { radius: 5.0 };
    let rect = Rectangle { width: 3.0, height: 4.0 };

    print_area(&circle);
    print_area(&rect);

    // Heterogeneous collection — different types in one Vec!
    // This is only possible with trait objects
    let shapes: Vec<Box<dyn Shape>> = vec![
        Box::new(Circle { radius: 1.0 }),
        Box::new(Rectangle { width: 2.0, height: 3.0 }),
        Box::new(Circle { radius: 4.0 }),
    ];

    let total_area: f64 = shapes.iter().map(|s| s.area()).sum();
    println!("Total area: {:.2}", total_area);
}
```

### 8.1 When to Use Which

| Feature | Generics (Static) | Trait Objects (Dynamic) |
|---------|-------------------|------------------------|
| Performance | Faster (no indirection) | Slight overhead (vtable lookup) |
| Binary size | Larger (code duplication) | Smaller (one copy) |
| Heterogeneous collections | No | Yes |
| Known at compile time | Yes (monomorphized) | No (runtime polymorphism) |
| Can use associated types | Yes | Limited |

**Rule of thumb:** Start with generics. Use `dyn Trait` when you need heterogeneous collections or when the concrete type is not known until runtime (e.g., plugin systems, GUI frameworks).

---

## 9. Supertraits and Trait Inheritance

A supertrait is a trait that requires another trait to be implemented first:

```rust
use std::fmt;

// Printable requires Display — it's a supertrait relationship
// Any type implementing Printable MUST also implement Display
trait Printable: fmt::Display {
    fn print(&self) {
        println!("{}", self); // we can use {} because Display is guaranteed
    }

    fn print_heading(&self) {
        println!("=== {} ===", self);
    }
}

struct Document {
    title: String,
    content: String,
}

// Must implement Display first (the supertrait)
impl fmt::Display for Document {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.title, self.content)
    }
}

// Then we can implement Printable
impl Printable for Document {}
// Uses default implementations of print() and print_heading()

fn main() {
    let doc = Document {
        title: String::from("Rust Guide"),
        content: String::from("Traits are powerful!"),
    };

    doc.print();         // "Rust Guide: Traits are powerful!"
    doc.print_heading(); // "=== Rust Guide: Traits are powerful! ==="
}
```

### 9.1 Multiple Supertraits

```rust
use std::fmt::{Debug, Display};

// Requires BOTH Debug and Display
trait Loggable: Debug + Display {
    fn log(&self) {
        // Debug for log files (structured), Display for users (human-readable)
        println!("[LOG] Display: {} | Debug: {:?}", self, self);
    }
}

#[derive(Debug)]
struct Event {
    name: String,
    severity: u8,
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Event '{}' (severity: {})", self.name, self.severity)
    }
}

impl Loggable for Event {}

fn main() {
    let event = Event {
        name: String::from("disk_full"),
        severity: 9,
    };
    event.log();
    // [LOG] Display: Event 'disk_full' (severity: 9) | Debug: Event { name: "disk_full", severity: 9 }
}
```

---

## 10. Practice Problems

### Problem 1: Area Trait
Define a `Measurable` trait with methods `area(&self) -> f64` and `perimeter(&self) -> f64`. Implement it for `Circle`, `Rectangle`, and `Triangle`. Write a generic function `largest_area<T: Measurable>(shapes: &[T]) -> f64` that returns the largest area in a slice.

### Problem 2: From Conversions
Create a `Temperature` enum with variants `Celsius(f64)`, `Fahrenheit(f64)`, and `Kelvin(f64)`. Implement `From<Celsius> for Fahrenheit`, `From<Fahrenheit> for Celsius`, and similar conversions. Also implement `Display` to print temperatures with their unit symbol.

### Problem 3: Custom Iterator
Create a `FibIterator` struct that implements `Iterator<Item = u64>`. It should yield Fibonacci numbers indefinitely. Use it with iterator methods: collect the first 10 Fibonacci numbers, find the first one greater than 1000, and compute the sum of the first 20.

### Problem 4: Plugin System with Trait Objects
Design a `Plugin` trait with methods `name(&self) -> &str` and `execute(&self, input: &str) -> String`. Create three different plugin structs (e.g., `UppercasePlugin`, `ReversePlugin`, `CensorPlugin`). Write a `PluginRunner` that stores `Vec<Box<dyn Plugin>>` and runs all plugins on a given input in sequence.

### Problem 5: Generic Stack
Implement a generic `Stack<T>` with methods `push`, `pop -> Option<T>`, `peek -> Option<&T>`, `is_empty`, and `size`. Then add a method `min() -> Option<&T>` with a trait bound `where T: Ord`. Write tests that use the stack with `i32`, `String`, and a custom struct.

---

## References

- [The Rust Programming Language, Ch. 10: Generic Types, Traits, and Lifetimes](https://doc.rust-lang.org/book/ch10-00-generics.html)
- [Rust by Example: Traits](https://doc.rust-lang.org/rust-by-example/trait.html)
- [Rust by Example: Generics](https://doc.rust-lang.org/rust-by-example/generics.html)
- [The Rust Reference: Trait Objects](https://doc.rust-lang.org/reference/types/trait-object.html)
- [std::fmt::Display](https://doc.rust-lang.org/std/fmt/trait.Display.html)
- [std::convert::From](https://doc.rust-lang.org/std/convert/trait.From.html)

---

**Previous**: [Error Handling](./09_Error_Handling.md) | **Next**: [Lifetimes](./11_Lifetimes.md)
