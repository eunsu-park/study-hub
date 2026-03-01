// 03_ownership.rs — Demonstrating Rust's ownership system
//
// Run: rustc 03_ownership.rs && ./03_ownership

fn main() {
    println!("=== Move Semantics ===");
    move_semantics();

    println!("\n=== Copy vs Move ===");
    copy_vs_move();

    println!("\n=== Ownership and Functions ===");
    ownership_and_functions();

    println!("\n=== Returning Ownership ===");
    returning_ownership();
}

/// Demonstrates how assignment moves heap-allocated data
fn move_semantics() {
    let s1 = String::from("hello");
    let s2 = s1; // s1 is MOVED to s2 — s1 is now invalid

    // Uncommenting the next line would cause a compile error:
    // println!("{s1}"); // ERROR: value borrowed here after move

    println!("s2 = {s2}");

    // Clone creates an independent deep copy
    let s3 = s2.clone();
    println!("s2 = {s2}, s3 = {s3}"); // Both valid — separate heap allocations
}

/// Shows which types are Copy (stack-only) vs Move (heap-involved)
fn copy_vs_move() {
    // Copy types — assignment duplicates the value
    let x: i32 = 42;
    let y = x; // Copy, not move
    println!("x = {x}, y = {y}"); // Both valid

    let a: (i32, bool) = (10, true);
    let b = a; // Tuple of Copy types is also Copy
    println!("a = {a:?}, b = {b:?}");

    // Move types — assignment transfers ownership
    let v1 = vec![1, 2, 3];
    let v2 = v1; // Move — v1 is now invalid
    // println!("{v1:?}"); // ERROR
    println!("v2 = {v2:?}");

    // A tuple containing a non-Copy type is NOT Copy
    let t1 = (42, String::from("hello"));
    let t2 = t1; // Move (because String is not Copy)
    // println!("{:?}", t1); // ERROR
    println!("t2 = {t2:?}");
}

/// Shows how passing values to functions transfers or copies ownership
fn ownership_and_functions() {
    let greeting = String::from("hello");
    takes_ownership(greeting);
    // greeting is no longer valid here — it was moved into the function

    let number = 42;
    makes_copy(number);
    println!("number is still valid: {number}"); // OK — i32 is Copy
}

fn takes_ownership(s: String) {
    println!("Took ownership of: {s}");
} // s is dropped here — heap memory freed

fn makes_copy(n: i32) {
    println!("Received copy: {n}");
} // n goes out of scope — nothing special happens (stack value)

/// Shows how functions can return ownership to the caller
fn returning_ownership() {
    let s1 = gives_ownership();
    println!("Got ownership: {s1}");

    let s2 = String::from("world");
    let s3 = takes_and_gives_back(s2);
    // s2 is now invalid; s3 owns the data
    println!("Got it back: {s3}");
}

fn gives_ownership() -> String {
    String::from("yours now") // Ownership moves to caller
}

fn takes_and_gives_back(s: String) -> String {
    println!("Borrowing briefly: {s}");
    s // Return ownership
}
