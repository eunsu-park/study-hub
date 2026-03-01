// Exercise: Borrowing and References
// Fix borrow checker errors and implement functions using references.
//
// Run: rustc 04_borrowing.rs && ./04_borrowing

fn main() {
    // Exercise 1: Fix the borrow checker error
    let mut s = String::from("hello");
    let r1 = &s;
    // let r2 = &mut s; // TODO: Fix this conflict
    // println!("{r1}, {r2}");

    // Exercise 2: Implement make_uppercase
    let mut greeting = String::from("hello world");
    make_uppercase(&mut greeting);
    println!("{greeting}"); // Should print "HELLO WORLD"

    // Exercise 3: Implement count_char (read-only borrow)
    let text = "mississippi";
    let count = count_char(text, 's');
    println!("'{text}' has {count} 's' characters");
    assert_eq!(count, 4);
    println!("text still valid: {text}"); // Proves we didn't take ownership

    // Exercise 4: Implement longest_word
    let sentence = "the quick brown fox jumps over the lazy dog";
    let longest = longest_word(sentence);
    println!("Longest word: '{longest}'");
    assert_eq!(longest, "jumps");

    println!("\nAll exercises passed!");
}

fn make_uppercase(s: &mut String) {
    // TODO: Convert the string to uppercase in-place
    // Hint: use s.make_ascii_uppercase() or build a new uppercase string
    todo!()
}

fn count_char(s: &str, c: char) -> usize {
    // TODO: Count occurrences of c in s without taking ownership
    todo!()
}

fn longest_word(s: &str) -> &str {
    // TODO: Return the longest word in the string
    // Hint: use split_whitespace() and track the longest
    todo!()
}
