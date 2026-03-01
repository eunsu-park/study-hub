// Exercise: Ownership
// Predict which lines compile, fix errors, and implement functions.
//
// Run: rustc 03_ownership.rs && ./03_ownership

fn main() {
    // Exercise 1: Predict and fix
    // Which println! lines will fail? Fix them without using .clone()
    let a = String::from("hello");
    let b = a;
    // println!("{a}"); // Will this compile? Fix if not.
    println!("{b}");

    // Exercise 2: Implement double and exclaim
    let x = 21;
    let doubled = double(x);
    println!("{x} doubled = {doubled}"); // x should still be valid

    let s = String::from("wow");
    let excited = exclaim(s);
    // println!("{s}"); // TODO: Why doesn't this work?
    println!("{excited}");

    // Exercise 3: First word extraction
    let sentence = String::from("hello world foo bar");
    let (first, rest) = first_word(sentence);
    println!("First: '{first}', Rest: '{rest}'");

    // Exercise 4: Custom Copy type
    let color = Color { r: 255, g: 128, b: 0 };
    print_color(color);
    println!("Color still valid: ({}, {}, {})", color.r, color.g, color.b);

    println!("\nAll exercises passed!");
}

fn double(x: i32) -> i32 {
    // TODO: Return x * 2
    todo!()
}

fn exclaim(s: String) -> String {
    // TODO: Append "!" to s and return it
    todo!()
}

fn first_word(s: String) -> (String, String) {
    // TODO: Split s at the first space, return (first_word, rest)
    // If no space, return (s, empty string)
    todo!()
}

// TODO: Add the right derives to make Color copyable
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

fn print_color(c: Color) {
    println!("Color: ({}, {}, {})", c.r, c.g, c.b);
}
