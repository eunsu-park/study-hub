// 10_traits_generics.rs — Traits, generics, and trait objects
//
// Run: rustc 10_traits_generics.rs && ./10_traits_generics

use std::fmt;

// Defining a trait
trait Summary {
    fn summarize(&self) -> String;

    // Default implementation — can be overridden
    fn preview(&self) -> String {
        format!("{}...", &self.summarize()[..20.min(self.summarize().len())])
    }
}

#[derive(Debug)]
struct Article {
    title: String,
    author: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{} by {}", self.title, self.author)
    }
}

#[derive(Debug)]
struct Tweet {
    username: String,
    text: String,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("@{}: {}", self.username, self.text)
    }
}

// Generic function with trait bound
fn print_summary<T: Summary>(item: &T) {
    println!("Summary: {}", item.summarize());
}

// impl Trait syntax (syntactic sugar for the above)
fn print_summary_alt(item: &impl Summary) {
    println!("Summary (alt): {}", item.summarize());
}

// Returning impl Trait
fn create_summarizable() -> impl Summary {
    Tweet {
        username: "rustlang".to_string(),
        text: "Rust 1.82 released!".to_string(),
    }
}

// Multiple trait bounds with where clause
fn display_and_summarize<T>(item: &T)
where
    T: Summary + fmt::Debug,
{
    println!("Debug: {item:?}");
    println!("Summary: {}", item.summarize());
}

// Generic struct
#[derive(Debug)]
struct Pair<T> {
    first: T,
    second: T,
}

impl<T: fmt::Display + PartialOrd> Pair<T> {
    fn larger(&self) -> &T {
        if self.first >= self.second {
            &self.first
        } else {
            &self.second
        }
    }
}

// Trait objects for dynamic dispatch
fn print_all_summaries(items: &[&dyn Summary]) {
    for item in items {
        println!("  - {}", item.summarize());
    }
}

// Implementing standard library traits
#[derive(Debug, Clone)]
struct Temperature {
    celsius: f64,
}

impl Temperature {
    fn new(celsius: f64) -> Self {
        Self { celsius }
    }
}

impl fmt::Display for Temperature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}°C", self.celsius)
    }
}

impl From<f64> for Temperature {
    fn from(celsius: f64) -> Self {
        Self { celsius }
    }
}

impl PartialEq for Temperature {
    fn eq(&self, other: &Self) -> bool {
        (self.celsius - other.celsius).abs() < f64::EPSILON
    }
}

fn main() {
    println!("=== Trait Implementations ===");
    let article = Article {
        title: "Rust in Production".to_string(),
        author: "Jane Doe".to_string(),
        content: "Rust is being adopted by major companies...".to_string(),
    };
    let tweet = Tweet {
        username: "ferris".to_string(),
        text: "I love Rust! 🦀".to_string(),
    };

    print_summary(&article);
    print_summary(&tweet);
    print_summary_alt(&article);

    println!("\n=== impl Trait Return ===");
    let item = create_summarizable();
    println!("{}", item.summarize());

    println!("\n=== Where Clause ===");
    display_and_summarize(&article);

    println!("\n=== Generic Struct ===");
    let pair = Pair {
        first: 10,
        second: 25,
    };
    println!("Larger of {}: {}", format!("({}, {})", pair.first, pair.second), pair.larger());

    println!("\n=== Trait Objects (dyn) ===");
    let items: Vec<&dyn Summary> = vec![&article, &tweet];
    print_all_summaries(&items);

    println!("\n=== Standard Traits ===");
    let t1 = Temperature::new(100.0);
    let t2: Temperature = 100.0.into(); // From trait
    println!("Display: {t1}");
    println!("Debug: {t1:?}");
    println!("Equal: {}", t1 == t2);

    let t3 = t1.clone();
    println!("Cloned: {t3}");
}
