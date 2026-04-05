// Exercise: Traits and Generics
// Implement traits, generics, and trait objects.
//
// Run: rustc 10_traits.rs && ./10_traits

use std::fmt;

fn main() {
    // Exercise 1: Implement Printable trait
    let article = Article { title: "Rust 2024".to_string(), author: "Ferris".to_string() };
    let tweet = Tweet { user: "rustlang".to_string(), body: "Hello!".to_string() };
    print_item(&article);
    print_item(&tweet);

    // Exercise 2: Generic max function
    assert_eq!(max_of_three(1, 3, 2), 3);
    assert_eq!(max_of_three(1.5, 3.7, 2.1), 3.7);
    assert_eq!(max_of_three("a", "c", "b"), "c");
    println!("Exercise 2 passed!");

    // Exercise 3: Trait objects
    let items: Vec<Box<dyn Printable>> = vec![Box::new(article), Box::new(tweet)];
    print_all(&items);

    // Exercise 4: Display implementation
    let p = Point { x: 3.0, y: 4.0 };
    println!("Point: {p}"); // Should print "(3.0, 4.0)"
    println!("Distance from origin: {:.2}", p.distance_from_origin());

    println!("\nAll exercises passed!");
}

// Exercise 1: Define the Printable trait and implement it
trait Printable {
    fn summary(&self) -> String;
}

struct Article { title: String, author: String }
struct Tweet { user: String, body: String }

// TODO: Implement Printable for Article and Tweet

fn print_item(item: &impl Printable) {
    println!("Summary: {}", item.summary());
}

// Exercise 2: Generic max function
fn max_of_three<T: PartialOrd>(a: T, b: T, c: T) -> T {
    // TODO: Return the maximum of three values
    todo!()
}

// Exercise 3: Print all trait objects
fn print_all(items: &[Box<dyn Printable>]) {
    // TODO: Print the summary of each item
    todo!()
}

// Exercise 4: Implement Display for Point
struct Point { x: f64, y: f64 }

impl Point {
    fn distance_from_origin(&self) -> f64 {
        // TODO
        todo!()
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Format as "(x, y)"
        todo!()
    }
}
