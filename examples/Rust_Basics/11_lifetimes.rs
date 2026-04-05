// 11_lifetimes.rs — Lifetime annotations and patterns
//
// Run: rustc 11_lifetimes.rs && ./11_lifetimes

// The classic longest() function — needs lifetime annotation because
// the compiler can't determine which input the return value borrows from
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() >= y.len() {
        x
    } else {
        y
    }
}

// Lifetime in struct — the struct cannot outlive the reference it holds
#[derive(Debug)]
struct Excerpt<'a> {
    text: &'a str,
}

impl<'a> Excerpt<'a> {
    // Lifetime elision: &self → return borrows from self
    fn first_word(&self) -> &str {
        self.text.split_whitespace().next().unwrap_or("")
    }

    // Explicit: return could borrow from self OR announcement
    fn announce(&self, announcement: &str) -> &str {
        println!("Attention: {announcement}");
        self.text
    }
}

// Multiple lifetimes — when references have different scopes
fn first_or_default<'a, 'b>(opt: Option<&'a str>, default: &'b str) -> &'b str
where
    'a: 'b, // 'a outlives 'b
{
    match opt {
        Some(s) => s,   // Safe because 'a: 'b guarantees s lives long enough
        None => default,
    }
}

// 'static lifetime — lives for the entire program
fn static_example() -> &'static str {
    "I'm a string literal — I live forever" // String literals are 'static
}

// Generic type with lifetime bounds
fn longest_with_announcement<'a, T>(x: &'a str, y: &'a str, ann: T) -> &'a str
where
    T: std::fmt::Display,
{
    println!("Announcement: {ann}");
    if x.len() >= y.len() {
        x
    } else {
        y
    }
}

fn main() {
    println!("=== Longest Function ===");
    let string1 = String::from("long string");
    let result;
    {
        let string2 = String::from("xyz");
        result = longest(string1.as_str(), string2.as_str());
        println!("Longest: {result}");
    }
    // result cannot be used here because string2 was dropped
    // The lifetime 'a is the intersection of both input lifetimes

    println!("\n=== Struct with Lifetime ===");
    let novel = String::from("Call me Ishmael. Some years ago...");
    let excerpt = Excerpt {
        text: &novel[..16], // Borrows from novel
    };
    println!("Excerpt: {:?}", excerpt);
    println!("First word: {}", excerpt.first_word());
    println!("Announce: {}", excerpt.announce("New excerpt"));

    println!("\n=== 'static Lifetime ===");
    let s = static_example();
    println!("{s}");

    println!("\n=== Lifetime with Generics ===");
    let x = "hello";
    let y = "world!";
    let result = longest_with_announcement(x, y, "Comparing strings");
    println!("Longest: {result}");

    println!("\n=== Lifetime Elision in Practice ===");
    // These function signatures have lifetimes elided by the compiler:
    // fn first_word(s: &str) -> &str
    //   becomes: fn first_word<'a>(s: &'a str) -> &'a str
    //
    // fn method(&self, s: &str) -> &str
    //   becomes: fn method<'a, 'b>(&'a self, s: &'b str) -> &'a str
    //   (return lifetime = &self lifetime)

    let words = "Rust is great for systems programming";
    let first = first_word_demo(words);
    println!("First word of \"{words}\": \"{first}\"");
}

// Elision rule 1: each &parameter gets its own lifetime
// Elision rule 2: if exactly one input lifetime, output gets it
// Elision rule 3: if &self is a parameter, output gets self's lifetime
fn first_word_demo(s: &str) -> &str {
    // Compiler adds: fn first_word_demo<'a>(s: &'a str) -> &'a str
    s.split_whitespace().next().unwrap_or("")
}
