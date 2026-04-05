// Exercise: Declarative Macros
// Practice writing macro_rules! macros with repetition and multiple arms.
//
// Run: rustc 20_declarative_macros.rs && ./20_declarative_macros

// Exercise 1: Hash set macro
// Create a macro that builds a HashSet from a comma-separated list.
macro_rules! hash_set {
    // TODO: Implement this macro
    ( $( $elem:expr ),* $(,)? ) => {
        {
            let mut s = std::collections::HashSet::new();
            $( s.insert($elem); )*
            s
        }
    };
}

// Exercise 2: Variadic min macro
// Write a min! macro that works with 2 or more arguments.
macro_rules! min {
    ($a:expr, $b:expr) => {
        if $a < $b { $a } else { $b }
    };
    ($a:expr, $b:expr, $($rest:expr),+) => {
        min!( min!($a, $b), $($rest),+ )
    };
}

// Exercise 3: Struct with Display
// TODO: Write a macro that generates a struct and auto-implements Display.
macro_rules! displayable_struct {
    (
        struct $name:ident {
            $( $field:ident : $ty:ty ),* $(,)?
        }
    ) => {
        #[derive(Debug)]
        struct $name {
            $( $field: $ty, )*
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{} {{ ", stringify!($name))?;
                $(
                    write!(f, "{}: {:?}, ", stringify!($field), self.$field)?;
                )*
                write!(f, "}}")
            }
        }
    };
}

// Exercise 4: Config DSL
// TODO: Write a config! macro that creates a HashMap<&str, HashMap<&str, String>>
// Syntax: config! { section [name] { key: value, ... } ... }

fn main() {
    use std::collections::HashSet;

    // Test Exercise 1
    let fruits = hash_set!["apple", "banana", "cherry"];
    assert!(fruits.contains("apple"));
    assert!(fruits.contains("banana"));
    assert_eq!(fruits.len(), 3);
    println!("hash_set! works: {fruits:?}");

    let empty: HashSet<i32> = hash_set![];
    assert!(empty.is_empty());
    println!("Empty hash_set! works");

    // Test Exercise 2
    assert_eq!(min!(5, 3), 3);
    assert_eq!(min!(10, 20, 5, 15), 5);
    assert_eq!(min!(1, 2, 3, 4, 5), 1);
    println!("min! works: min!(10, 20, 5, 15) = {}", min!(10, 20, 5, 15));

    // Test Exercise 3
    displayable_struct! {
        struct Point {
            x: f64,
            y: f64,
        }
    }

    let p = Point { x: 3.0, y: 4.0 };
    println!("Display: {p}");

    // Exercise 5: JSON-like DSL (challenge)
    // TODO: Write a json! macro that produces nested data structures
    // json!({ "name": "Alice", "age": 30, "scores": [90, 85] })

    println!("\nAll exercises passed!");
}
