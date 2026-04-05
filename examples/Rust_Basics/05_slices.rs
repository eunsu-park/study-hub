// 05_slices.rs — String slices, array slices, and slice methods
//
// Run: rustc 05_slices.rs && ./05_slices

fn main() {
    println!("=== String Slices ===");
    string_slices();

    println!("\n=== String vs &str ===");
    string_vs_str();

    println!("\n=== Array and Vec Slices ===");
    array_slices();

    println!("\n=== Slice Methods ===");
    slice_methods();

    println!("\n=== Functions Taking Slices ===");
    slice_functions();
}

fn string_slices() {
    let s = String::from("hello world");

    // Range syntax for slicing
    let hello = &s[0..5];   // "hello"
    let world = &s[6..11];  // "world"
    println!("{hello}, {world}");

    // Shorthand ranges
    let start = &s[..5];    // From beginning
    let end = &s[6..];      // To end
    let full = &s[..];      // Entire string
    println!("start={start}, end={end}, full={full}");

    // String literals are already &str
    let literal: &str = "I am a string slice";
    println!("{literal}");
}

fn string_vs_str() {
    // String — owned, heap-allocated, growable
    let mut owned = String::from("hello");
    owned.push_str(", world");
    owned.push('!');
    println!("String: {owned}");

    // &str — borrowed view, cannot be modified
    let slice: &str = &owned;
    println!("&str: {slice}");

    // Common conversions
    let from_literal: String = "hello".to_string();
    let from_method: String = String::from("hello");
    let back_to_str: &str = &from_literal;
    println!("{from_method}, {back_to_str}");

    // Collecting from chars
    let filtered: String = "h3ll0 w0rld"
        .chars()
        .filter(|c| c.is_alphabetic())
        .collect();
    println!("Filtered: {filtered}");
}

fn array_slices() {
    let arr = [10, 20, 30, 40, 50];

    // Slice of an array
    let middle: &[i32] = &arr[1..4]; // [20, 30, 40]
    println!("Middle: {middle:?}");

    // Vec slices work the same way
    let v = vec![1, 2, 3, 4, 5, 6];
    let first_three: &[i32] = &v[..3];
    let last_two: &[i32] = &v[v.len() - 2..];
    println!("First three: {first_three:?}");
    println!("Last two: {last_two:?}");

    // Mutable slices
    let mut data = [5, 3, 1, 4, 2];
    let slice = &mut data[..];
    slice.sort();
    println!("Sorted: {data:?}");
}

fn slice_methods() {
    let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let s: &[i32] = &numbers;

    println!("len: {}", s.len());
    println!("is_empty: {}", s.is_empty());
    println!("first: {:?}", s.first());
    println!("last: {:?}", s.last());
    println!("contains 5: {}", s.contains(&5));

    // Splitting
    let (left, right) = s.split_at(5);
    println!("split_at(5): {left:?} | {right:?}");

    // Chunks
    print!("chunks(3): ");
    for chunk in s.chunks(3) {
        print!("{chunk:?} ");
    }
    println!();

    // Windows (sliding)
    print!("windows(3): ");
    for w in s.windows(3) {
        print!("{w:?} ");
    }
    println!();

    // String slice methods
    let text = "hello, world, rust";
    let parts: Vec<&str> = text.split(", ").collect();
    println!("Split: {parts:?}");
    println!("starts_with: {}", text.starts_with("hello"));
    println!("trim: \"  spaces  \".trim() = \"{}\"", "  spaces  ".trim());
}

/// Accept &str instead of String for maximum flexibility
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b' ' {
            return &s[..i];
        }
    }
    s
}

/// Accept &[i32] instead of &Vec<i32> for flexibility
fn sum_slice(values: &[i32]) -> i32 {
    values.iter().sum()
}

fn slice_functions() {
    // first_word works with both String and &str
    let owned = String::from("hello world");
    println!("first_word(String): {}", first_word(&owned));
    println!("first_word(&str): {}", first_word("rust programming"));

    // sum_slice works with both arrays and Vecs
    let arr = [1, 2, 3];
    let vec = vec![4, 5, 6];
    println!("sum array: {}", sum_slice(&arr));
    println!("sum vec: {}", sum_slice(&vec));
    println!("sum partial: {}", sum_slice(&vec[1..]));
}
