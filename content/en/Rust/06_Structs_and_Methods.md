# 06. Structs and Methods

**Previous**: [Slices](./05_Slices.md) | **Next**: [Enums and Pattern Matching](./07_Enums_and_Pattern_Matching.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define structs with named fields, tuple structs, and unit structs
2. Implement methods and associated functions using `impl` blocks
3. Use `#[derive]` to automatically implement common traits
4. Apply struct update syntax for creating modified copies
5. Understand how struct ownership interacts with borrowing

---

Structs are Rust's primary tool for creating custom data types. Combined with `impl` blocks, they provide the same encapsulation as classes in OOP languages — but without inheritance. Rust favors **composition over inheritance**, using traits (Lesson 10) for polymorphism instead.

## Table of Contents
1. [Defining Structs](#1-defining-structs)
2. [Methods and Associated Functions](#2-methods-and-associated-functions)
3. [Derived Traits](#3-derived-traits)
4. [Struct Patterns](#4-struct-patterns)
5. [Practice Problems](#5-practice-problems)

---

## 1. Defining Structs

### 1.1 Named-Field Structs

```rust
struct User {
    username: String,
    email: String,
    active: bool,
    sign_in_count: u64,
}

fn main() {
    // Creating an instance
    let user1 = User {
        username: String::from("alice"),
        email: String::from("alice@example.com"),
        active: true,
        sign_in_count: 1,
    };

    // Accessing fields
    println!("{}", user1.username);

    // Mutable struct — the entire struct must be mut
    let mut user2 = User {
        username: String::from("bob"),
        email: String::from("bob@example.com"),
        active: true,
        sign_in_count: 0,
    };
    user2.sign_in_count += 1;
}
```

### 1.2 Field Init Shorthand and Update Syntax

```rust
fn build_user(username: String, email: String) -> User {
    User {
        username,        // Shorthand: field name matches parameter name
        email,
        active: true,
        sign_in_count: 1,
    }
}

fn main() {
    let user1 = build_user(String::from("alice"), String::from("alice@ex.com"));

    // Struct update syntax — create from existing, overriding some fields
    let user2 = User {
        email: String::from("bob@ex.com"),
        ..user1  // Remaining fields from user1
    };
    // Note: user1.username was MOVED into user2
    // user1.email was overridden, but user1.username is now invalid
    // user1.active and user1.sign_in_count are Copy, so they're fine
}
```

### 1.3 Tuple Structs

```rust
// Named tuples — useful for type distinction
struct Color(u8, u8, u8);
struct Point(f64, f64, f64);

fn main() {
    let red = Color(255, 0, 0);
    let origin = Point(0.0, 0.0, 0.0);

    // Access by index
    println!("R={}, G={}, B={}", red.0, red.1, red.2);

    // Destructuring
    let Point(x, y, z) = origin;
    println!("({x}, {y}, {z})");
}
```

### 1.4 Unit Structs

```rust
// No fields — useful as markers or for trait implementations
struct AlwaysEqual;

fn main() {
    let _subject = AlwaysEqual;
}
```

---

## 2. Methods and Associated Functions

### 2.1 Defining Methods

```rust
#[derive(Debug)]
struct Rectangle {
    width: f64,
    height: f64,
}

impl Rectangle {
    // Method: takes &self (immutable borrow of the instance)
    fn area(&self) -> f64 {
        self.width * self.height
    }

    // Method with mutable self
    fn scale(&mut self, factor: f64) {
        self.width *= factor;
        self.height *= factor;
    }

    // Method that takes ownership (rare)
    fn into_square(self) -> Rectangle {
        let side = self.width.max(self.height);
        Rectangle { width: side, height: side }
    }
}

fn main() {
    let mut rect = Rectangle { width: 30.0, height: 50.0 };
    println!("Area: {}", rect.area());

    rect.scale(2.0);
    println!("Scaled: {rect:?}");

    let square = rect.into_square();  // rect is consumed
    println!("Square: {square:?}");
}
```

### 2.2 Associated Functions (Constructors)

```rust
impl Rectangle {
    // Associated function — no self parameter, called with ::
    fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    fn square(size: f64) -> Self {
        Self { width: size, height: size }
    }
}

fn main() {
    let rect = Rectangle::new(10.0, 20.0);
    let sq = Rectangle::square(15.0);
}
```

### 2.3 Multiple impl Blocks

```rust
// You can split methods across multiple impl blocks
// Useful for organizing code or conditional compilation
impl Rectangle {
    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }
}

impl Rectangle {
    fn is_square(&self) -> bool {
        (self.width - self.height).abs() < f64::EPSILON
    }
}
```

### 2.4 Method Call Syntax and Auto-referencing

```rust
fn main() {
    let rect = Rectangle::new(10.0, 20.0);

    // These are equivalent — Rust adds &, &mut, or * automatically
    rect.area();       // Auto-borrows: (&rect).area()
    (&rect).area();    // Explicit borrow

    // This auto-referencing is why Rust doesn't have -> (like C++)
    // The compiler determines &self, &mut self, or self from the method signature
}
```

---

## 3. Derived Traits

```rust
#[derive(Debug, Clone, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

fn main() {
    let p1 = Point { x: 1.0, y: 2.0 };

    // Debug — enables {:?} formatting
    println!("{p1:?}");           // Point { x: 1.0, y: 2.0 }
    println!("{p1:#?}");          // Pretty-printed

    // Clone — explicit deep copy
    let p2 = p1.clone();

    // PartialEq — enables == and !=
    assert_eq!(p1, p2);
}
```

Common derivable traits:

| Trait | Purpose | Enables |
|-------|---------|---------|
| `Debug` | Debug formatting | `{:?}`, `{:#?}` |
| `Clone` | Explicit deep copy | `.clone()` |
| `Copy` | Implicit bitwise copy | Assignment without move |
| `PartialEq` | Equality comparison | `==`, `!=` |
| `Eq` | Total equality (no NaN) | Required by HashMap keys |
| `Hash` | Hashing | HashMap/HashSet keys |
| `Default` | Default value | `Type::default()` |
| `PartialOrd` | Partial ordering | `<`, `>`, `<=`, `>=` |
| `Ord` | Total ordering | `.sort()` on collections |

---

## 4. Struct Patterns

### 4.1 Builder Pattern

```rust
#[derive(Debug)]
struct Config {
    host: String,
    port: u16,
    max_connections: u32,
    timeout_secs: u64,
}

impl Config {
    fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }
}

#[derive(Default)]
struct ConfigBuilder {
    host: String,
    port: u16,
    max_connections: u32,
    timeout_secs: u64,
}

impl ConfigBuilder {
    fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    fn max_connections(mut self, n: u32) -> Self {
        self.max_connections = n;
        self
    }

    fn build(self) -> Config {
        Config {
            host: if self.host.is_empty() { "localhost".to_string() } else { self.host },
            port: if self.port == 0 { 8080 } else { self.port },
            max_connections: if self.max_connections == 0 { 100 } else { self.max_connections },
            timeout_secs: if self.timeout_secs == 0 { 30 } else { self.timeout_secs },
        }
    }
}

fn main() {
    let config = Config::builder()
        .host("example.com")
        .port(3000)
        .max_connections(500)
        .build();
    println!("{config:#?}");
}
```

### 4.2 Newtype Pattern

```rust
// Wrap a primitive to create a distinct type with domain meaning
struct Meters(f64);
struct Seconds(f64);

impl Meters {
    fn new(value: f64) -> Self {
        Self(value)
    }

    fn value(&self) -> f64 {
        self.0
    }
}

fn speed(distance: Meters, time: Seconds) -> f64 {
    distance.0 / time.0
}

fn main() {
    let d = Meters::new(100.0);
    let t = Seconds(9.58);
    // Cannot accidentally pass Seconds as Meters — the type system prevents it
    println!("Speed: {:.2} m/s", speed(d, t));
}
```

---

## 5. Practice Problems

### Exercise 1: Circle
Define a `Circle` struct with `radius: f64`. Implement methods `area()`, `circumference()`, and an associated function `new(radius: f64) -> Self`.

### Exercise 2: Student Record
Define a `Student` struct with `name: String`, `grades: Vec<f64>`. Implement `average()`, `highest()`, `is_passing(threshold: f64) -> bool`.

### Exercise 3: Struct Update Syntax
Create a `ServerConfig` struct. Demonstrate creating a production config from a development config using struct update syntax, and explain which fields are moved vs copied.

### Exercise 4: Newtype Enforcement
Create `Celsius(f64)` and `Fahrenheit(f64)` newtypes. Implement conversion methods between them. Show that the compiler prevents mixing them up.

### Exercise 5: Builder Implementation
Implement a builder pattern for an `Email` struct with `from`, `to`, `subject`, and `body` fields. The builder should validate that `from` and `to` are non-empty before building.

---

## References
- [The Rust Book — Structs](https://doc.rust-lang.org/book/ch05-00-structs.html)
- [Rust by Example — Structures](https://doc.rust-lang.org/rust-by-example/custom_types/structs.html)

---

**Previous**: [Slices](./05_Slices.md) | **Next**: [Enums and Pattern Matching](./07_Enums_and_Pattern_Matching.md)
