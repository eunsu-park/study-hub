// 08_collections.rs — Vec, HashMap, and iterator chaining
//
// Run: rustc 08_collections.rs && ./08_collections

use std::collections::HashMap;

fn main() {
    println!("=== Vec<T> ===");
    vec_demo();

    println!("\n=== String ===");
    string_demo();

    println!("\n=== HashMap ===");
    hashmap_demo();

    println!("\n=== Iterator Chaining ===");
    iterator_demo();
}

fn vec_demo() {
    // Creating vectors
    let mut numbers: Vec<i32> = Vec::new();
    numbers.push(10);
    numbers.push(20);
    numbers.push(30);

    let squares = vec![1, 4, 9, 16, 25]; // vec! macro

    // Indexing (panics on out-of-bounds)
    println!("First: {}", numbers[0]);

    // Safe access with .get() returns Option
    match numbers.get(10) {
        Some(val) => println!("Found: {val}"),
        None => println!("Index 10 out of bounds"),
    }

    // Iteration
    for n in &numbers {
        print!("{n} ");
    }
    println!();

    // Mutation during iteration
    let mut data = vec![1, 2, 3, 4, 5];
    for n in &mut data {
        *n *= 2;
    }
    println!("Doubled: {data:?}");

    // Useful methods
    println!("squares contains 16: {}", squares.contains(&16));
    println!("len={}, capacity={}", numbers.len(), numbers.capacity());

    // Sorting and dedup
    let mut mixed = vec![3, 1, 4, 1, 5, 9, 2, 6, 5];
    mixed.sort();
    mixed.dedup();
    println!("Sorted + deduped: {mixed:?}");
}

fn string_demo() {
    // String (owned, heap-allocated) vs &str (borrowed slice)
    let mut s = String::from("Hello");
    s.push(' ');
    s.push_str("World");
    println!("{s}");

    // String concatenation
    let hello = String::from("Hello");
    let world = String::from("World");
    let combined = format!("{hello}, {world}!"); // Preferred — no ownership issues
    println!("{combined}");

    // Iterating over characters (not bytes!)
    let emoji_str = "Hello 🦀!";
    println!("Chars: {:?}", emoji_str.chars().collect::<Vec<_>>());
    println!("Bytes: {} chars: {}", emoji_str.len(), emoji_str.chars().count());

    // Splitting and collecting
    let csv = "alice,bob,charlie";
    let names: Vec<&str> = csv.split(',').collect();
    println!("Names: {names:?}");
}

fn hashmap_demo() {
    let mut scores: HashMap<String, i32> = HashMap::new();

    // Insert
    scores.insert("Alice".to_string(), 95);
    scores.insert("Bob".to_string(), 87);
    scores.insert("Charlie".to_string(), 92);

    // Access
    if let Some(score) = scores.get("Alice") {
        println!("Alice's score: {score}");
    }

    // Entry API — insert only if key doesn't exist
    scores.entry("Alice".to_string()).or_insert(0); // Keeps 95
    scores.entry("Dave".to_string()).or_insert(78); // Inserts 78

    // Entry API — modify existing value
    let text = "hello world hello rust hello world";
    let mut word_count: HashMap<&str, u32> = HashMap::new();
    for word in text.split_whitespace() {
        let count = word_count.entry(word).or_insert(0);
        *count += 1;
    }
    println!("Word counts: {word_count:?}");

    // Iteration
    println!("All scores:");
    for (name, score) in &scores {
        println!("  {name}: {score}");
    }
}

fn iterator_demo() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // Filter + map + collect
    let even_squares: Vec<i32> = numbers
        .iter()
        .filter(|&&n| n % 2 == 0)
        .map(|&n| n * n)
        .collect();
    println!("Even squares: {even_squares:?}");

    // Sum and fold
    let sum: i32 = numbers.iter().sum();
    let product: i32 = numbers.iter().fold(1, |acc, &n| acc * n);
    println!("Sum: {sum}, Product: {product}");

    // Enumerate
    let fruits = vec!["apple", "banana", "cherry"];
    for (i, fruit) in fruits.iter().enumerate() {
        println!("  {i}: {fruit}");
    }

    // Zip two iterators
    let names = vec!["Alice", "Bob", "Charlie"];
    let ages = vec![30, 25, 35];
    let people: Vec<_> = names.iter().zip(ages.iter()).collect();
    println!("Zipped: {people:?}");

    // find and position
    let first_even = numbers.iter().find(|&&n| n % 2 == 0);
    let pos = numbers.iter().position(|&n| n > 5);
    println!("First even: {first_even:?}, Position > 5: {pos:?}");
}
