# 20. Declarative Macros

**Previous**: [Build System Deep Dive](./03_Build_System.md) | **Next**: [Procedural Macros](./05_Procedural_Macros.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write declarative macros using `macro_rules!` with multiple match arms
2. Use repetition operators (`*`, `+`, `?`) to match variable-length input
3. Understand fragment specifiers (`expr`, `ty`, `ident`, `tt`, etc.)
4. Debug macro expansion with `cargo expand` and `trace_macros!`
5. Apply practical macro patterns for code generation, DSLs, and boilerplate reduction

---

Rust's macro system is one of its most powerful features. Unlike C/C++ text-substitution macros, Rust macros operate on the **abstract syntax tree (AST)** — they are hygienic, type-aware, and checked at compile time. This lesson covers **declarative macros** (also called "macros by example"), defined with `macro_rules!`.

## Table of Contents
1. [Why Macros?](#1-why-macros)
2. [macro_rules! Basics](#2-macro_rules-basics)
3. [Fragment Specifiers](#3-fragment-specifiers)
4. [Repetition](#4-repetition)
5. [Multiple Match Arms](#5-multiple-match-arms)
6. [Practical Patterns](#6-practical-patterns)
7. [Macro Hygiene](#7-macro-hygiene)
8. [Debugging Macros](#8-debugging-macros)
9. [Scoping and Exporting](#9-scoping-and-exporting)
10. [Common Pitfalls](#10-common-pitfalls)
11. [Real-World Examples](#11-real-world-examples)
12. [Exercises](#12-exercises)

---

## 1. Why Macros?

Functions cannot do everything. Consider these limitations:

```rust
// You can't write a function that accepts a variable number of arguments
// println! is a macro because of this
println!("one: {}", 1);
println!("two: {} {}", 1, 2);
println!("three: {} {} {}", 1, 2, 3);

// You can't write a function that generates struct definitions
// or implements traits automatically

// You can't write a function that creates new identifiers
```

Macros fill these gaps. They run at **compile time**, generating code before type checking and borrow checking occur.

### Functions vs Macros

| Feature | Function | Macro |
|---------|----------|-------|
| Evaluation time | Runtime | Compile time |
| Variable arguments | No (use slices) | Yes |
| Code generation | No | Yes |
| Hygiene | N/A | Yes (scoped) |
| Type checking | On the function | On the *expanded* code |
| Debugging | Standard | Requires expansion tools |

---

## 2. macro_rules! Basics

A declarative macro matches patterns against input tokens and produces replacement code:

```rust
// Simplest possible macro — no arguments
macro_rules! say_hello {
    () => {
        println!("Hello from a macro!");
    };
}

fn main() {
    say_hello!();  // Expands to: println!("Hello from a macro!");
}
```

### Invocation Styles

Macros can be invoked with parentheses, brackets, or braces:

```rust
macro_rules! my_macro {
    () => { 42 };
}

fn main() {
    let a = my_macro!();   // Parentheses — most common for expression-like macros
    let b = my_macro![];   // Brackets — conventional for vec![], array-like macros
    let c = my_macro!{};   // Braces — conventional for item-defining macros
    assert_eq!(a, b);
    assert_eq!(b, c);
}
```

Convention: use `()` for function-like invocations, `[]` for literal-like invocations (`vec![]`), and `{}` for item-level macros.

### Capturing Arguments

Use `$name:specifier` to capture input tokens:

```rust
macro_rules! create_greeting {
    ($name:expr) => {
        format!("Hello, {}!", $name)
    };
}

fn main() {
    let greeting = create_greeting!("Rust");
    println!("{greeting}");  // Hello, Rust!

    let user = String::from("Alice");
    let greeting = create_greeting!(user);
    println!("{greeting}");  // Hello, Alice!
}
```

---

## 3. Fragment Specifiers

Fragment specifiers tell the macro parser what kind of syntax to expect:

| Specifier | Matches | Example |
|-----------|---------|---------|
| `expr` | Any expression | `1 + 2`, `foo()`, `if x { 1 } else { 2 }` |
| `ty` | A type | `i32`, `Vec<String>`, `&'a str` |
| `ident` | An identifier | `foo`, `MyStruct`, `x` |
| `pat` | A pattern | `Some(x)`, `(a, b)`, `_` |
| `path` | A path | `std::io::Result`, `crate::module` |
| `stmt` | A statement | `let x = 1`, `x += 1` |
| `block` | A block `{ ... }` | `{ let x = 1; x + 1 }` |
| `item` | An item | `fn foo() {}`, `struct Bar;` |
| `meta` | Attribute content | `derive(Debug)`, `cfg(test)` |
| `tt` | A single token tree | Any single token or `(...)` / `[...]` / `{...}` group |
| `literal` | A literal value | `42`, `"hello"`, `true` |
| `lifetime` | A lifetime | `'a`, `'static` |
| `vis` | Visibility qualifier | `pub`, `pub(crate)`, (empty) |

### Using Type Specifiers

```rust
macro_rules! declare_pair {
    ($name:ident, $t:ty) => {
        struct $name {
            first: $t,
            second: $t,
        }
    };
}

declare_pair!(IntPair, i32);
declare_pair!(StringPair, String);

fn main() {
    let pair = IntPair { first: 1, second: 2 };
    println!("Pair: ({}, {})", pair.first, pair.second);

    let sp = StringPair {
        first: "hello".into(),
        second: "world".into(),
    };
    println!("Pair: ({}, {})", sp.first, sp.second);
}
```

### The `tt` Specifier (Token Tree)

`tt` is the most flexible specifier — it matches any single token or a balanced group of tokens in delimiters:

```rust
macro_rules! apply {
    ($func:ident, $($arg:tt)*) => {
        $func($($arg)*)
    };
}

fn add(a: i32, b: i32) -> i32 { a + b }

fn main() {
    let result = apply!(add, 3, 4);
    println!("apply!(add, 3, 4) = {result}");  // 7
}
```

---

## 4. Repetition

Repetition is what makes macros truly powerful. The syntax is `$(...) separator repetition_operator`:

- `*` — zero or more
- `+` — one or more
- `?` — zero or one

```rust
// vec![] clone — create a Vec from a list of elements
macro_rules! my_vec {
    // Match a comma-separated list of expressions
    ( $( $element:expr ),* ) => {
        {
            let mut v = Vec::new();
            $( v.push($element); )*
            v
        }
    };
    // Also handle trailing comma
    ( $( $element:expr ),+ , ) => {
        my_vec![ $( $element ),* ]
    };
}

fn main() {
    let v = my_vec![1, 2, 3, 4, 5];
    println!("{v:?}");  // [1, 2, 3, 4, 5]

    let v = my_vec!["hello", "world",];  // Trailing comma OK
    println!("{v:?}");  // ["hello", "world"]
}
```

### Nested Repetition

```rust
// Create a HashMap from key => value pairs
macro_rules! hash_map {
    ( $( $key:expr => $value:expr ),* $(,)? ) => {
        {
            let mut map = std::collections::HashMap::new();
            $( map.insert($key, $value); )*
            map
        }
    };
}

fn main() {
    let scores = hash_map! {
        "Alice" => 95,
        "Bob" => 87,
        "Charlie" => 92,
    };

    for (name, score) in &scores {
        println!("{name}: {score}");
    }
}
```

### Repetition with Multiple Bindings

When you use multiple captures in a repetition, they must repeat the same number of times:

```rust
macro_rules! named_values {
    ( $( $name:ident = $value:expr ),* $(,)? ) => {
        $(
            let $name = $value;
            println!("{} = {}", stringify!($name), $name);
        )*
    };
}

fn main() {
    named_values! {
        x = 10,
        y = 20,
        z = 30,
    }
    // Prints:
    // x = 10
    // y = 20
    // z = 30

    println!("Sum: {}", x + y + z);  // 60
}
```

---

## 5. Multiple Match Arms

Like `match` expressions, macros can have multiple arms. The macro tries each arm in order:

```rust
macro_rules! calculate {
    // Single value — identity
    ($x:expr) => { $x };

    // Two values with an operator
    ($x:expr, +, $y:expr) => { $x + $y };
    ($x:expr, -, $y:expr) => { $x - $y };
    ($x:expr, *, $y:expr) => { $x * $y };
    ($x:expr, /, $y:expr) => { $x / $y };
}

fn main() {
    println!("{}", calculate!(5));           // 5
    println!("{}", calculate!(10, +, 20));   // 30
    println!("{}", calculate!(100, /, 4));   // 25
}
```

### Overloading by Shape

```rust
macro_rules! log_message {
    // No arguments — just a separator line
    () => {
        println!("---");
    };

    // Just a message
    ($msg:expr) => {
        println!("[LOG] {}", $msg);
    };

    // Message with a level
    ($level:ident, $msg:expr) => {
        println!("[{:>5}] {}", stringify!($level).to_uppercase(), $msg);
    };

    // Message with level and key-value context
    ($level:ident, $msg:expr, $( $key:ident = $val:expr ),+ ) => {
        print!("[{:>5}] {}", stringify!($level).to_uppercase(), $msg);
        $( print!(" {}={}", stringify!($key), $val); )+
        println!();
    };
}

fn main() {
    log_message!();
    log_message!("Application started");
    log_message!(info, "Request received");
    log_message!(error, "Connection failed", host = "db.example.com", retries = 3);
}
```

Output:
```
---
[LOG] Application started
[ INFO] Request received
[ERROR] Connection failed host=db.example.com retries=3
```

---

## 6. Practical Patterns

### Pattern 1: Builder Macro

```rust
// Requires: cargo add paste
macro_rules! builder {
    (
        $name:ident {
            $( $field:ident : $ty:ty ),* $(,)?
        }
    ) => {
        #[derive(Debug, Clone)]
        struct $name {
            $( $field: $ty, )*
        }

        paste::paste! {  // Requires the `paste` crate for identifier manipulation
            struct [<$name Builder>] {
                $( $field: Option<$ty>, )*
            }

            impl $name {
                fn builder() -> [<$name Builder>] {
                    [<$name Builder>] {
                        $( $field: None, )*
                    }
                }
            }
        }
    };
}

// A more practical version without external crates:
macro_rules! make_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $( $field_vis:vis $field:ident : $ty:ty ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis struct $name {
            $( $field_vis $field: $ty, )*
        }

        impl $name {
            $vis fn new( $( $field: $ty ),* ) -> Self {
                Self { $( $field, )* }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}(", stringify!($name))?;
                let fields = vec![
                    $( format!("{}: {:?}", stringify!($field), self.$field), )*
                ];
                write!(f, "{})", fields.join(", "))
            }
        }
    };
}

make_struct! {
    #[derive(Debug, Clone)]
    pub struct Config {
        pub host: String,
        pub port: u16,
        pub debug: bool,
    }
}

fn main() {
    let config = Config::new("localhost".into(), 8080, true);
    println!("{config}");  // Config(host: "localhost", port: 8080, debug: true)
}
```

### Pattern 2: Enum with Methods

```rust
macro_rules! enum_with_str {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $variant:ident => $str:literal ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $( $variant, )*
        }

        impl $name {
            pub fn as_str(&self) -> &'static str {
                match self {
                    $( $name::$variant => $str, )*
                }
            }

            pub fn from_str(s: &str) -> Option<Self> {
                match s {
                    $( $str => Some($name::$variant), )*
                    _ => None,
                }
            }

            pub fn all() -> &'static [Self] {
                &[ $( $name::$variant, )* ]
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.as_str())
            }
        }
    };
}

enum_with_str! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Color {
        Red => "red",
        Green => "green",
        Blue => "blue",
        Yellow => "yellow",
    }
}

fn main() {
    let c = Color::Red;
    println!("Color: {c}");                    // red
    println!("From str: {:?}", Color::from_str("blue"));  // Some(Blue)
    println!("All: {:?}", Color::all());       // [Red, Green, Blue, Yellow]
}
```

### Pattern 3: Test Generation

```rust
macro_rules! test_cases {
    ( $func:ident : $( ($input:expr) => $expected:expr );+ $(;)? ) => {
        $(
            paste::paste! {
                #[test]
                fn [< test_ $func _ $input >]() {
                    assert_eq!($func($input), $expected);
                }
            }
        )+
    };
}

// Simpler version without paste:
macro_rules! test_suite {
    ($name:ident, $func:expr, $( ($input:expr, $expected:expr) ),+ $(,)? ) => {
        #[cfg(test)]
        mod $name {
            use super::*;

            $(
                #[test]
                fn test() {
                    assert_eq!($func($input), $expected,
                        "Failed for input: {:?}", $input);
                }
            )+
        }
    };
}

fn double(n: i32) -> i32 { n * 2 }

fn is_even(n: i32) -> bool { n % 2 == 0 }

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_doubles {
        ( $( $input:expr => $expected:expr ),* $(,)? ) => {
            $(
                assert_eq!(double($input), $expected,
                    "double({}) should be {}", $input, $expected);
            )*
        };
    }

    #[test]
    fn test_double() {
        assert_doubles! {
            0 => 0,
            1 => 2,
            5 => 10,
            -3 => -6,
        }
    }
}
```

---

## 7. Macro Hygiene

Rust macros are **hygienic** — variables defined inside a macro don't leak into the caller's scope, and the caller's variables don't accidentally shadow macro internals:

```rust
macro_rules! using_x {
    ($body:expr) => {
        {
            let x = 42;  // This 'x' is in the macro's scope
            $body         // This uses the caller's 'x', not the macro's
        }
    };
}

fn main() {
    let x = 10;
    // This prints the caller's x (10), not the macro's x (42)
    // because of hygiene. The macro's `let x = 42` creates a
    // different binding that doesn't conflict.
    let result = using_x!(x + 1);
    println!("result: {result}");  // 11, not 43

    // However, if we don't reference an outer 'x':
    let result = using_x!({
        let y = 100;
        y
    });
    println!("result: {result}");  // 100
}
```

### Breaking Hygiene (When Needed)

Sometimes you intentionally want a macro to introduce bindings visible to the caller. The idiomatic approach is to let the caller name the variable:

```rust
macro_rules! let_binding {
    ($name:ident = $value:expr) => {
        let $name = $value;
    };
}

fn main() {
    let_binding!(x = 42);
    println!("x = {x}");  // 42 — works because caller chose the name
}
```

---

## 8. Debugging Macros

### cargo expand

The most useful tool for debugging macros. Install and use:

```bash
cargo install cargo-expand

# Expand all macros in your crate
cargo expand

# Expand macros in a specific module
cargo expand module_name

# Expand a specific function
cargo expand main
```

### trace_macros! (Nightly Only)

```rust
#![feature(trace_macros)]

macro_rules! my_add {
    ($a:expr, $b:expr) => { $a + $b };
}

fn main() {
    trace_macros!(true);
    let x = my_add!(1, 2);
    trace_macros!(false);
    println!("{x}");
}
```

Output during compilation:
```
note: trace_macro
  --> src/main.rs:8:13
   |
8  |     let x = my_add!(1, 2);
   |             ^^^^^^^^^^^^^^
   |
   = note: expanding `my_add! { 1, 2 }`
   = note: to `1 + 2`
```

### stringify! for Inspection

```rust
macro_rules! debug_expand {
    ($($tokens:tt)*) => {
        println!("Input tokens: {}", stringify!($($tokens)*));
        $($tokens)*
    };
}

fn main() {
    debug_expand! {
        let x = 1 + 2;
        println!("x = {x}");
    }
    // Prints:
    // Input tokens: let x = 1 + 2 ; println! ("x = {x}") ;
    // x = 3
}
```

### Compile Error Messages

Use `compile_error!` to produce clear errors from macros:

```rust
macro_rules! validated_enum {
    ( $name:ident { $( $variant:ident ),+ $(,)? } ) => {
        enum $name { $( $variant, )+ }
    };
    ( $name:ident { } ) => {
        compile_error!("Enum must have at least one variant");
    };
}

validated_enum!(Direction { North, South, East, West });
// validated_enum!(Empty {});  // Compile error: "Enum must have at least one variant"
```

---

## 9. Scoping and Exporting

### Module-Level Visibility

By default, macros defined with `macro_rules!` are scoped to the module where they're defined. They must be defined **before** use in the same file:

```rust
// This works — macro defined before use
macro_rules! greet {
    () => { println!("Hello!") };
}
greet!();

// To use a macro across modules in the same crate, use #[macro_export]
#[macro_export]
macro_rules! exported_macro {
    () => { println!("I'm available everywhere!") };
}
```

### #[macro_export]

`#[macro_export]` makes the macro available at the crate root:

```rust
// In lib.rs or any module
#[macro_export]
macro_rules! my_assert {
    ($cond:expr) => {
        if !$cond {
            panic!("Assertion failed: {}", stringify!($cond));
        }
    };
    ($cond:expr, $msg:expr) => {
        if !$cond {
            panic!("Assertion failed: {} — {}", stringify!($cond), $msg);
        }
    };
}

// Users of the crate can do:
// use my_crate::my_assert;
```

### #[macro_use]

For internal module organization:

```rust
// In macros.rs
#[macro_export]
macro_rules! helper {
    () => {};
}

// In main.rs — pull macros from a module
#[macro_use]
mod macros;

// Now helper!() is available here
```

---

## 10. Common Pitfalls

### Pitfall 1: Operator Precedence

```rust
macro_rules! double {
    ($x:expr) => { $x * 2 };  // Looks fine...
}

fn main() {
    println!("{}", double!(3 + 1));  // 8, not 7
    // Expands to: (3 + 1) * 2 = 8
    // Because $x:expr captures the full expression "3 + 1"
    // This is actually CORRECT in Rust, unlike C macros!
}

// In C, #define DOUBLE(x) x * 2
// DOUBLE(3 + 1) expands to 3 + 1 * 2 = 5  (wrong!)
// Rust's hygiene handles this correctly.
```

### Pitfall 2: Matching Ambiguity

```rust
// BAD: These arms are ambiguous for input like `foo, bar`
// macro_rules! ambiguous {
//     ($($a:expr),*) => { "list" };
//     ($a:expr, $b:expr) => { "pair" };
// }
// The first arm always matches, so the second is unreachable.

// FIX: Put more specific arms first
macro_rules! fixed {
    ($a:expr, $b:expr) => { "pair" };
    ($($a:expr),*) => { "list" };
}
```

### Pitfall 3: Recursive Expansion Limits

```rust
macro_rules! count {
    () => { 0usize };
    ($head:tt $($tail:tt)*) => { 1usize + count!($($tail)*) };
}

fn main() {
    let n = count!(a b c d e);
    println!("Count: {n}");  // 5
}

// Default recursion limit is 128. For deeply recursive macros:
// #![recursion_limit = "256"]
```

### Pitfall 4: Type Inference in Repeated Blocks

```rust
macro_rules! make_vec_bad {
    ($($elem:expr),*) => {
        {
            let mut v = Vec::new();
            $( v.push($elem); )*
            v
        }
    };
}

// This might fail type inference if elements have different apparent types
// let v = make_vec_bad!(1, 2u8, 3);  // Error: mismatched types

// Fix: let the caller specify the type
macro_rules! typed_vec {
    ($t:ty; $($elem:expr),* $(,)?) => {
        {
            let mut v: Vec<$t> = Vec::new();
            $( v.push($elem as $t); )*
            v
        }
    };
}
```

---

## 11. Real-World Examples

### Mini assert_eq! Implementation

```rust
macro_rules! my_assert_eq {
    ($left:expr, $right:expr) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                if *left_val != *right_val {
                    panic!(
                        "assertion failed: `(left == right)`\n  left: `{:?}`\n right: `{:?}`",
                        left_val, right_val
                    );
                }
            }
        }
    };
    ($left:expr, $right:expr, $($msg:tt)+) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                if *left_val != *right_val {
                    panic!(
                        "assertion failed: `(left == right)`\n  left: `{:?}`\n right: `{:?}`\n  note: {}",
                        left_val, right_val, format!($($msg)+)
                    );
                }
            }
        }
    };
}

fn main() {
    my_assert_eq!(1 + 1, 2);
    my_assert_eq!("hello".len(), 5, "string length mismatch");
    println!("All assertions passed!");
}
```

### Configuration DSL

```rust
use std::collections::HashMap;

macro_rules! config {
    (
        $( section [$section:ident] {
            $( $key:ident : $value:expr ),* $(,)?
        } )*
    ) => {
        {
            let mut sections: HashMap<&str, HashMap<&str, String>> = HashMap::new();
            $(
                let mut section_map = HashMap::new();
                $( section_map.insert(stringify!($key), format!("{}", $value)); )*
                sections.insert(stringify!($section), section_map);
            )*
            sections
        }
    };
}

fn main() {
    let cfg = config! {
        section [database] {
            host: "localhost",
            port: 5432,
            name: "mydb",
        }
        section [server] {
            host: "0.0.0.0",
            port: 8080,
            workers: 4,
        }
    };

    for (section, values) in &cfg {
        println!("[{section}]");
        for (key, value) in values {
            println!("  {key} = {value}");
        }
    }
}
```

### Retry Macro

```rust
use std::time::Duration;
use std::thread;

macro_rules! retry {
    ($attempts:expr, $delay_ms:expr, $body:expr) => {{
        let mut last_err = None;
        for attempt in 1..=$attempts {
            match $body {
                Ok(val) => {
                    return Ok(val);
                }
                Err(e) => {
                    eprintln!("Attempt {attempt}/{}: {e}", $attempts);
                    last_err = Some(e);
                    if attempt < $attempts {
                        thread::sleep(Duration::from_millis($delay_ms));
                    }
                }
            }
        }
        Err(last_err.unwrap())
    }};
}

fn flaky_operation(counter: &std::sync::atomic::AtomicU32) -> Result<String, String> {
    use std::sync::atomic::Ordering;
    let count = counter.fetch_add(1, Ordering::SeqCst);
    if count < 2 {
        Err(format!("Temporary failure #{}", count + 1))
    } else {
        Ok("Success!".into())
    }
}

fn do_work() -> Result<String, String> {
    let counter = std::sync::atomic::AtomicU32::new(0);
    retry!(5, 100, flaky_operation(&counter))
}

fn main() {
    match do_work() {
        Ok(msg) => println!("Result: {msg}"),
        Err(e) => println!("All attempts failed: {e}"),
    }
}
```

---

## 12. Exercises

1. **Hash set macro**: Write `hash_set!` that creates a `HashSet` from a comma-separated list of elements, similar to `vec![]`.

2. **min/max variadic**: Write a `min!` macro that accepts 2+ arguments and returns the minimum. Example: `min!(5, 3, 8, 1)` returns `1`.

3. **Struct with Display**: Write a macro that generates a struct definition and automatically implements `Display` by printing each field name and value.

4. **Retry with backoff**: Extend the `retry!` macro to support exponential backoff (double the delay each attempt).

5. **JSON-like DSL**: Write a `json!` macro that produces a nested structure from JSON-like syntax: `json!({ "name": "Alice", "age": 30, "scores": [90, 85, 92] })`.

---

## References

- [The Rust Reference: Macros by Example](https://doc.rust-lang.org/reference/macros-by-example.html)
- [The Little Book of Rust Macros](https://veykril.github.io/tlborm/)
- [Rust by Example: Macros](https://doc.rust-lang.org/rust-by-example/macros.html)
- [cargo-expand](https://github.com/dtolnay/cargo-expand)

---

**Previous**: [Build System Deep Dive](./03_Build_System.md) | **Next**: [Procedural Macros](./05_Procedural_Macros.md)
