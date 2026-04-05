// 20_declarative_macros.rs — macro_rules! patterns and repetition
//
// Run: rustc 20_declarative_macros.rs && ./20_declarative_macros

use std::collections::HashMap;

// HashMap literal macro
macro_rules! hash_map {
    ( $( $key:expr => $value:expr ),* $(,)? ) => {
        {
            let mut map = HashMap::new();
            $( map.insert($key, $value); )*
            map
        }
    };
}

// Variadic min macro
macro_rules! min {
    ($a:expr) => { $a };
    ($a:expr, $($rest:expr),+) => {
        {
            let a = $a;
            let b = min!($($rest),+);
            if a < b { a } else { b }
        }
    };
}

// Enum with string conversion
macro_rules! string_enum {
    ( $name:ident { $( $variant:ident => $str:literal ),* $(,)? } ) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
        enum $name { $( $variant, )* }

        impl $name {
            fn as_str(&self) -> &'static str {
                match self { $( $name::$variant => $str, )* }
            }
            fn from_str(s: &str) -> Option<Self> {
                match s { $( $str => Some($name::$variant), )* _ => None }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.as_str())
            }
        }
    };
}

string_enum!(Direction {
    North => "north",
    South => "south",
    East => "east",
    West => "west",
});

// Logging macro with levels
macro_rules! log {
    (info, $($arg:tt)*) => { println!("[INFO] {}", format!($($arg)*)); };
    (warn, $($arg:tt)*) => { println!("[WARN] {}", format!($($arg)*)); };
    (error, $($arg:tt)*) => { println!("[ERROR] {}", format!($($arg)*)); };
}

fn main() {
    println!("=== HashMap Macro ===");
    let scores = hash_map! {
        "Alice" => 95,
        "Bob" => 87,
        "Charlie" => 92,
    };
    for (name, score) in &scores {
        println!("  {name}: {score}");
    }

    println!("\n=== Min Macro ===");
    println!("  min!(5, 3) = {}", min!(5, 3));
    println!("  min!(10, 20, 5, 15) = {}", min!(10, 20, 5, 15));
    println!("  min!(1) = {}", min!(1));

    println!("\n=== String Enum ===");
    let dir = Direction::North;
    println!("  Direction: {dir}");
    println!("  From str: {:?}", Direction::from_str("east"));
    println!("  From str: {:?}", Direction::from_str("invalid"));

    println!("\n=== Log Macro ===");
    log!(info, "Server started on port {}", 8080);
    log!(warn, "High memory usage: {}%", 85);
    log!(error, "Connection to {} failed", "database");
}
