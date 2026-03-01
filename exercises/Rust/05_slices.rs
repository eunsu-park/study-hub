// Exercise: Slices
// Implement functions using string and array slices.
//
// Run: rustc 05_slices.rs && ./05_slices

fn main() {
    // Exercise 1: First word
    assert_eq!(first_word("hello world"), "hello");
    assert_eq!(first_word("single"), "single");
    assert_eq!(first_word(""), "");
    println!("Exercise 1 passed!");

    // Exercise 2: Reverse words
    assert_eq!(reverse_words("hello world"), "world hello");
    assert_eq!(reverse_words("one"), "one");
    println!("Exercise 2 passed!");

    // Exercise 3: Moving average
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let avg = moving_average(&data, 3);
    assert_eq!(avg, vec![2.0, 3.0, 4.0]);
    println!("Exercise 3 passed!");

    // Exercise 4: Safe substring
    let s = "Hello, 🦀 Rust!";
    assert!(safe_substring(s, 0, 5).is_some());
    assert!(safe_substring(s, 0, 8).is_none()); // Falls inside emoji
    assert!(safe_substring(s, 0, 100).is_none()); // Out of bounds
    println!("Exercise 4 passed!");

    println!("\nAll exercises passed!");
}

fn first_word(s: &str) -> &str {
    // TODO: Return the first word (text before first space, or entire string)
    todo!()
}

fn reverse_words(s: &str) -> String {
    // TODO: Reverse the order of words
    todo!()
}

fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    // TODO: Compute moving average using data.windows(window)
    todo!()
}

fn safe_substring(s: &str, start: usize, end: usize) -> Option<&str> {
    // TODO: Return None if range is invalid or falls on non-UTF-8 boundary
    // Hint: use s.get(start..end)
    todo!()
}
