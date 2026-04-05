// Exercise: Lifetimes
// Add lifetime annotations to make the code compile.
//
// Run: rustc 11_lifetimes.rs && ./11_lifetimes

fn main() {
    // Exercise 1: Fix the longest function
    let s1 = String::from("long string");
    let result;
    {
        let s2 = String::from("xyz");
        result = longest(s1.as_str(), s2.as_str());
        println!("Longest: {result}");
    }

    // Exercise 2: Struct with lifetime
    let text = String::from("Call me Ishmael. Some years ago...");
    let excerpt = Excerpt { part: &text[..16] };
    println!("Excerpt: {}", excerpt.part);
    println!("First word: {}", excerpt.first_word());

    // Exercise 3: Multiple lifetimes
    let greeting = String::from("Hello");
    let name = String::from("World");
    let result = combine(&greeting, &name);
    println!("{result}");

    println!("\nAll exercises passed!");
}

// Exercise 1: Add lifetime annotations
// fn longest(x: &str, y: &str) -> &str {
//     if x.len() >= y.len() { x } else { y }
// }
fn longest(x: &str, y: &str) -> &str {
    // TODO: Add lifetime annotations to make this compile
    todo!()
}

// Exercise 2: Add lifetime to struct
struct Excerpt {
    part: &str, // TODO: Add lifetime annotation
}

impl Excerpt {
    fn first_word(&self) -> &str {
        // TODO: Return the first word of self.part
        todo!()
    }
}

// Exercise 3: Function that creates a new String — does it need lifetimes?
fn combine(a: &str, b: &str) -> String {
    // TODO: Return "{a}, {b}!" — explain why this doesn't need lifetime annotations
    todo!()
}
