// Exercise: Collections
// Practice with Vec, HashMap, and iterator chaining.
//
// Run: rustc 08_collections.rs && ./08_collections

use std::collections::HashMap;

fn main() {
    // Exercise 1: Vector statistics
    let data = vec![4, 7, 2, 9, 1, 5, 8, 3, 6, 10];
    let stats = vec_stats(&data);
    println!("Stats: min={}, max={}, sum={}, avg={:.1}",
        stats.0, stats.1, stats.2, stats.3);
    assert_eq!(stats, (1, 10, 55, 5.5));

    // Exercise 2: Word frequency counter
    let text = "the cat sat on the mat the cat";
    let freq = word_frequency(text);
    assert_eq!(freq.get("the"), Some(&3));
    assert_eq!(freq.get("cat"), Some(&2));
    println!("Word frequency passed!");

    // Exercise 3: Group by first letter
    let words = vec!["apple", "banana", "avocado", "blueberry", "cherry", "apricot"];
    let grouped = group_by_first_letter(&words);
    assert_eq!(grouped.get(&'a').unwrap().len(), 3);
    assert_eq!(grouped.get(&'b').unwrap().len(), 2);
    println!("Group by first letter passed!");

    // Exercise 4: Iterator chain — transform and filter
    let result = transform_filter(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    // Square even numbers, filter those > 20
    assert_eq!(result, vec![36, 64, 100]);
    println!("Iterator chain passed!");

    println!("\nAll exercises passed!");
}

fn vec_stats(data: &[i32]) -> (i32, i32, i32, f64) {
    // TODO: Return (min, max, sum, average)
    todo!()
}

fn word_frequency(text: &str) -> HashMap<&str, u32> {
    // TODO: Count word frequencies using the entry API
    todo!()
}

fn group_by_first_letter<'a>(words: &[&'a str]) -> HashMap<char, Vec<&'a str>> {
    // TODO: Group words by their first character
    todo!()
}

fn transform_filter(data: &[i32]) -> Vec<i32> {
    // TODO: Filter even numbers, square them, keep only those > 20
    todo!()
}
