// 02_variables_and_types.rs — Variables, mutability, shadowing, and types
//
// Run: rustc 02_variables_and_types.rs && ./02_variables_and_types

fn main() {
    println!("=== Immutability and Mutability ===");
    immutability_demo();

    println!("\n=== Shadowing ===");
    shadowing_demo();

    println!("\n=== Scalar Types ===");
    scalar_types();

    println!("\n=== Compound Types ===");
    compound_types();

    println!("\n=== Type Conversion ===");
    type_conversion();

    println!("\n=== Constants vs Let ===");
    constants_demo();
}

fn immutability_demo() {
    let x = 5;
    println!("Immutable x = {x}");
    // x = 6; // ERROR: cannot assign twice to immutable variable

    let mut counter = 0;
    counter += 1;
    counter += 1;
    println!("Mutable counter = {counter}");
}

fn shadowing_demo() {
    let x = 5;
    println!("Original x = {x}");

    let x = x + 1; // Shadow with new value
    println!("Shadowed x = {x}");

    let x = "now I'm a string"; // Shadow with different type
    println!("Rebound x = {x}");

    // Shadowing in inner scope
    {
        let x = x.len();
        println!("Inner x (length) = {x}");
    }
    println!("Outer x unchanged = {x}");
}

fn scalar_types() {
    // Integer types
    let byte: u8 = 255;
    let signed: i8 = -128;
    let default_int = 42; // i32 by default
    let big: i64 = 9_000_000_000;
    let arch: usize = 100; // Platform pointer size

    println!("u8: {byte}, i8: {signed}, i32: {default_int}, i64: {big}, usize: {arch}");

    // Integer literals
    let hex = 0xff;
    let octal = 0o77;
    let binary = 0b1111_0000;
    let byte_literal = b'A'; // u8 only
    println!("hex={hex}, octal={octal}, binary={binary}, byte={byte_literal}");

    // Floats
    let pi: f64 = 3.141_592_653_589_793;
    let approx: f32 = 3.14;
    println!("f64: {pi}, f32: {approx}");

    // Bool and char
    let is_rust_fun: bool = true;
    let emoji: char = '🦀'; // char is 4 bytes (Unicode scalar value)
    println!("bool: {is_rust_fun}, char: {emoji} (size: {} bytes)", std::mem::size_of::<char>());
}

fn compound_types() {
    // Tuple — fixed-size collection of different types
    let person: (&str, i32, bool) = ("Alice", 30, true);
    println!("Name: {}, Age: {}, Active: {}", person.0, person.1, person.2);

    // Destructuring
    let (name, age, _active) = person;
    println!("Destructured: {name}, {age}");

    // Array — fixed-size, same type, stack-allocated
    let months = ["Jan", "Feb", "Mar", "Apr", "May"];
    println!("First: {}, Last: {}", months[0], months[months.len() - 1]);

    // Array with repeated value
    let zeros = [0i32; 5];
    println!("Zeros: {zeros:?}");

    // Iterating
    print!("Months: ");
    for m in &months {
        print!("{m} ");
    }
    println!();
}

fn type_conversion() {
    // Numeric casting with `as`
    let x: i32 = 42;
    let y: f64 = x as f64;
    let z: u8 = x as u8;
    println!("i32 {x} → f64 {y}, u8 {z}");

    // Truncation (be careful)
    let big: i32 = 300;
    let truncated: u8 = big as u8; // 300 % 256 = 44
    println!("i32 {big} → u8 {truncated} (truncated!)");

    // String ↔ number
    let parsed: i32 = "123".parse().expect("Not a number");
    let back: String = parsed.to_string();
    println!("Parsed: {parsed}, Back to string: \"{back}\"");
}

const MAX_CONNECTIONS: u32 = 100;

fn constants_demo() {
    // const — must have type annotation, evaluated at compile time
    println!("MAX_CONNECTIONS = {MAX_CONNECTIONS}");

    // const can appear in any scope
    const TIMEOUT_SECS: u64 = 30;
    println!("TIMEOUT_SECS = {TIMEOUT_SECS}");

    // Unlike let, const can NEVER be mut or shadowed within the same scope
    // const values are inlined at each usage site by the compiler
}
