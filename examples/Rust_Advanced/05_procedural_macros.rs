// 05_procedural_macros.rs — Proc macro concepts and patterns
//
// Run: rustc 05_procedural_macros.rs && ./05_procedural_macros
//
// Note: Real procedural macros require a separate crate with proc-macro = true.
// This example demonstrates the concepts and patterns using declarative macros
// and trait-based approaches that mirror what proc macros generate.

use std::fmt;

fn main() {
    println!("=== Derive-like Pattern ===");
    derive_like_demo();

    println!("\n=== Attribute-like Pattern ===");
    attribute_like_demo();

    println!("\n=== Function-like Macro ===");
    function_like_demo();

    println!("\n=== Token-level Manipulation ===");
    token_manipulation();
}

// --- Derive-like: what #[derive(Debug, Serialize)] generates ---

// Simulating what a derive macro produces
trait Describe {
    fn describe(&self) -> String;
    fn field_names(&self) -> Vec<&'static str>;
}

struct User {
    name: String,
    age: u32,
    email: String,
}

// This is what #[derive(Describe)] would generate:
impl Describe for User {
    fn describe(&self) -> String {
        format!(
            "User {{ name: {:?}, age: {:?}, email: {:?} }}",
            self.name, self.age, self.email
        )
    }

    fn field_names(&self) -> Vec<&'static str> {
        vec!["name", "age", "email"]
    }
}

fn derive_like_demo() {
    let user = User {
        name: "Alice".into(),
        age: 30,
        email: "alice@example.com".into(),
    };
    println!("  {}", user.describe());
    println!("  Fields: {:?}", user.field_names());
}

// --- Attribute-like: what #[route(GET, "/api")] generates ---

struct Route {
    method: &'static str,
    path: &'static str,
    handler_name: &'static str,
    handler: fn() -> String,
}

fn index_handler() -> String {
    "Welcome to the API".to_string()
}

fn users_handler() -> String {
    r#"[{"name": "Alice"}, {"name": "Bob"}]"#.to_string()
}

fn health_handler() -> String {
    r#"{"status": "ok"}"#.to_string()
}

fn attribute_like_demo() {
    // What #[route(GET, "/api")] would register:
    let routes = vec![
        Route { method: "GET", path: "/", handler_name: "index_handler", handler: index_handler },
        Route { method: "GET", path: "/users", handler_name: "users_handler", handler: users_handler },
        Route { method: "GET", path: "/health", handler_name: "health_handler", handler: health_handler },
    ];

    for route in &routes {
        println!("  {} {} -> {}()", route.method, route.path, route.handler_name);
        println!("    Response: {}", (route.handler)());
    }
}

// --- Function-like macro ---

// Declarative macro that mimics a proc macro's token-level power
macro_rules! make_struct {
    ($name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        #[derive(Debug)]
        struct $name {
            $($field: $ty),*
        }

        impl $name {
            fn new($($field: $ty),*) -> Self {
                Self { $($field),* }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{} {{ ", stringify!($name))?;
                $(write!(f, "{}: {:?}, ", stringify!($field), self.$field)?;)*
                write!(f, "}}")
            }
        }
    };
}

make_struct!(Point { x: f64, y: f64, z: f64 });
make_struct!(Config { host: String, port: u16 });

fn function_like_demo() {
    let p = Point::new(1.0, 2.0, 3.0);
    println!("  Debug: {p:?}");
    println!("  Display: {p}");

    let c = Config::new("localhost".into(), 8080);
    println!("  {c}");
}

// --- Token manipulation concepts ---

macro_rules! count_args {
    () => { 0 };
    ($head:expr $(, $tail:expr)*) => { 1 + count_args!($($tail),*) };
}

macro_rules! hash_map {
    ($($key:expr => $value:expr),* $(,)?) => {{
        let mut map = std::collections::HashMap::new();
        $(map.insert($key, $value);)*
        map
    }};
}

macro_rules! assert_fields {
    ($struct_val:expr, { $($field:ident : $expected:expr),* $(,)? }) => {
        $(
            assert_eq!(
                $struct_val.$field, $expected,
                "Field `{}` mismatch: got {:?}, expected {:?}",
                stringify!($field), $struct_val.$field, $expected
            );
            println!("  ✓ {}.{} == {:?}", stringify!($struct_val), stringify!($field), $expected);
        )*
    };
}

fn token_manipulation() {
    // count_args!
    println!("  count_args!() = {}", count_args!());
    println!("  count_args!(a, b, c) = {}", count_args!("a", "b", "c"));

    // hash_map!
    let scores = hash_map! {
        "Alice" => 95,
        "Bob" => 87,
        "Charlie" => 92,
    };
    println!("  Scores: {scores:?}");

    // assert_fields!
    let p = Point::new(1.0, 2.0, 3.0);
    assert_fields!(p, { x: 1.0, y: 2.0, z: 3.0 });
}
