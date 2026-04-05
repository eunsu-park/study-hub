// 01_getting_started.rs — Hello World and basic Rust program structure
//
// Run: rustc 01_getting_started.rs && ./01_getting_started

fn main() {
    println!("=== Hello, World! ===");
    hello_world();

    println!("\n=== Formatted Output ===");
    formatted_output();

    println!("\n=== Basic Expressions ===");
    basic_expressions();

    println!("\n=== Control Flow Basics ===");
    control_flow_basics();
}

/// The classic first program
fn hello_world() {
    println!("Hello, world!");
    println!("Welcome to Rust!");
}

/// Demonstrates println! formatting options
fn formatted_output() {
    let name = "Rust";
    let version = 2024;

    // Positional placeholder
    println!("{name} edition {version}");

    // Debug formatting with {:?}
    println!("Debug: {:?}", (1, "hello", true));

    // Padding and alignment
    println!("{:<15} | left-aligned", "hello");
    println!("{:>15} | right-aligned", "hello");
    println!("{:^15} | centered", "hello");

    // Number formatting
    println!("Binary:  {:08b}", 42);
    println!("Hex:     {:#06x}", 255);
    println!("Float:   {:.3}", std::f64::consts::PI);
}

/// Shows that blocks are expressions in Rust
fn basic_expressions() {
    // Block expression — the last line (without semicolon) is the value
    let result = {
        let x = 5;
        let y = 10;
        x + y // No semicolon → this is the return value
    };
    println!("Block result: {result}");

    // if/else as an expression
    let temp = 35;
    let description = if temp > 30 { "hot" } else { "mild" };
    println!("{temp}°C is {description}");

    // match as an expression
    let code = 404;
    let meaning = match code {
        200 => "OK",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    println!("HTTP {code}: {meaning}");
}

/// Basic loops and control flow
fn control_flow_basics() {
    // loop with break returning a value
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 5 {
            break counter * 10;
        }
    };
    println!("loop result: {result}");

    // for range
    let mut sum = 0;
    for i in 1..=10 {
        sum += i;
    }
    println!("Sum 1..=10: {sum}");

    // while let
    let mut stack = vec![1, 2, 3];
    print!("Popping: ");
    while let Some(top) = stack.pop() {
        print!("{top} ");
    }
    println!();
}
