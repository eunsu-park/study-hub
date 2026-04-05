// 22_advanced_traits.rs — Trait objects, blanket impls, type-state pattern
//
// Run: rustc 22_advanced_traits.rs && ./22_advanced_traits

use std::fmt;
use std::marker::PhantomData;

// === Extension Trait (Blanket Impl) ===

trait StringExt {
    fn truncate_display(&self, max: usize) -> String;
    fn is_blank(&self) -> bool;
}

impl<T: AsRef<str>> StringExt for T {
    fn truncate_display(&self, max: usize) -> String {
        let s = self.as_ref();
        if s.len() <= max {
            s.to_string()
        } else {
            format!("{}...", &s[..max])
        }
    }

    fn is_blank(&self) -> bool {
        self.as_ref().trim().is_empty()
    }
}

// === Type-State Pattern ===

struct Draft;
struct Published;

struct Article<State> {
    title: String,
    body: String,
    _state: PhantomData<State>,
}

impl Article<Draft> {
    fn new(title: &str, body: &str) -> Self {
        Article {
            title: title.to_string(),
            body: body.to_string(),
            _state: PhantomData,
        }
    }

    fn edit(&mut self, body: &str) {
        self.body = body.to_string();
    }

    fn publish(self) -> Article<Published> {
        println!("Publishing: {}", self.title);
        Article {
            title: self.title,
            body: self.body,
            _state: PhantomData,
        }
    }
}

impl Article<Published> {
    fn read(&self) -> &str {
        &self.body
    }
    // Cannot edit a published article — no edit() method here
}

impl<S> fmt::Display for Article<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\" ({} chars)", self.title, self.body.len())
    }
}

// === Dynamic Dispatch with Trait Objects ===

trait Shape: fmt::Debug {
    fn area(&self) -> f64;
    fn perimeter(&self) -> f64;
}

#[derive(Debug)]
struct Circle { radius: f64 }
#[derive(Debug)]
struct Rectangle { width: f64, height: f64 }

impl Shape for Circle {
    fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius }
    fn perimeter(&self) -> f64 { 2.0 * std::f64::consts::PI * self.radius }
}

impl Shape for Rectangle {
    fn area(&self) -> f64 { self.width * self.height }
    fn perimeter(&self) -> f64 { 2.0 * (self.width + self.height) }
}

fn print_shapes(shapes: &[Box<dyn Shape>]) {
    for shape in shapes {
        println!("  {:?} — area: {:.2}, perimeter: {:.2}",
            shape, shape.area(), shape.perimeter());
    }
}

// === Newtype Pattern ===

struct Meters(f64);
struct Kilometers(f64);

impl From<Meters> for Kilometers {
    fn from(m: Meters) -> Self { Kilometers(m.0 / 1000.0) }
}

impl fmt::Display for Kilometers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} km", self.0)
    }
}

fn main() {
    println!("=== Extension Trait ===");
    println!("  Truncate: {}", "Hello, World!".truncate_display(5));
    println!("  Is blank: {}", "   ".is_blank());
    println!("  Is blank: {}", "hello".is_blank());

    println!("\n=== Type-State Pattern ===");
    let mut article = Article::<Draft>::new("Rust Traits", "Draft content...");
    article.edit("Revised content about traits.");
    println!("  Draft: {article}");

    let published = article.publish();
    println!("  Published: {published}");
    println!("  Content: {}", published.read());
    // published.edit("oops");  // Compile error! Can't edit published articles.

    println!("\n=== Trait Objects ===");
    let shapes: Vec<Box<dyn Shape>> = vec![
        Box::new(Circle { radius: 5.0 }),
        Box::new(Rectangle { width: 4.0, height: 6.0 }),
        Box::new(Circle { radius: 1.0 }),
    ];
    print_shapes(&shapes);

    println!("\n=== Newtype ===");
    let distance = Meters(42195.0);
    let km: Kilometers = distance.into();
    println!("  Marathon: {km}");
}
