// 04_borrowing.rs — References and borrowing rules
//
// Run: rustc 04_borrowing.rs && ./04_borrowing

fn main() {
    println!("=== Immutable References ===");
    immutable_refs();

    println!("\n=== Mutable References ===");
    mutable_refs();

    println!("\n=== Borrowing Rules ===");
    borrowing_rules();

    println!("\n=== Reference Patterns ===");
    reference_patterns();
}

/// Immutable references let you read without taking ownership
fn immutable_refs() {
    let s = String::from("hello");
    let len = calculate_length(&s); // Borrow s — no ownership transfer
    println!("'{s}' has length {len}"); // s is still valid!

    // Multiple immutable references are fine
    let r1 = &s;
    let r2 = &s;
    println!("r1={r1}, r2={r2}");
}

fn calculate_length(s: &String) -> usize {
    s.len()
    // s goes out of scope but doesn't own the String — nothing is dropped
}

/// Mutable references allow modification through a borrow
fn mutable_refs() {
    let mut s = String::from("hello");
    append_world(&mut s);
    println!("After append: {s}");

    // Swap two values using mutable references
    let mut a = 10;
    let mut b = 20;
    swap(&mut a, &mut b);
    println!("After swap: a={a}, b={b}");
}

fn append_world(s: &mut String) {
    s.push_str(", world!");
}

fn swap(a: &mut i32, b: &mut i32) {
    let temp = *a; // Dereference to get the value
    *a = *b;
    *b = temp;
}

/// Demonstrates the borrowing rules and Non-Lexical Lifetimes (NLL)
fn borrowing_rules() {
    let mut data = String::from("hello");

    // Rule: either ONE mutable ref OR any number of immutable refs
    let r1 = &data;
    let r2 = &data;
    println!("Immutable borrows: {r1}, {r2}");
    // r1 and r2 are no longer used after this point (NLL)

    // Now we can take a mutable reference — no conflict
    let r3 = &mut data;
    r3.push_str("!");
    println!("Mutable borrow: {r3}");

    // This pattern is common: read first, then modify
    let numbers = vec![1, 2, 3, 4, 5];
    let sum: i32 = numbers.iter().sum(); // Immutable borrow
    println!("Sum = {sum}");
    // numbers is still owned by this scope
}

/// Common patterns with references
fn reference_patterns() {
    // Pattern 1: Read-only access to a collection
    let items = vec!["apple", "banana", "cherry"];
    print_items(&items);
    println!("Items still accessible: {items:?}");

    // Pattern 2: Modify a collection through mutable reference
    let mut scores = vec![85, 92, 78, 95, 88];
    normalize(&mut scores, 100);
    println!("Normalized: {scores:?}");

    // Pattern 3: Choose between &T and T based on need
    let name = String::from("Alice");
    greet(&name);        // Borrow — name stays valid
    greet_owned(name);   // Move — name is consumed
    // println!("{name}"); // ERROR: name was moved
}

fn print_items(items: &[&str]) {
    for (i, item) in items.iter().enumerate() {
        println!("  {}: {item}", i + 1);
    }
}

fn normalize(scores: &mut Vec<i32>, max: i32) {
    for score in scores.iter_mut() {
        *score = (*score * 100) / max;
    }
}

fn greet(name: &str) {
    println!("Hello, {name}! (borrowed)");
}

fn greet_owned(name: String) {
    println!("Hello, {name}! (owned — I'll clean this up)");
}
