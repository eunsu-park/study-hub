// Exercise: Advanced Traits
// Practice trait objects, associated types, blanket impls, and sealed traits.
//
// Run: rustc 22_advanced_traits.rs && ./22_advanced_traits

use std::fmt;

// Exercise 1: Extension trait
// Add a `frequencies` method to all iterators that returns a HashMap
// of element counts.
trait IteratorExt: Iterator {
    fn frequencies(self) -> std::collections::HashMap<Self::Item, usize>
    where
        Self: Sized,
        Self::Item: std::hash::Hash + Eq,
    {
        // TODO: Implement this
        let mut map = std::collections::HashMap::new();
        for item in self {
            *map.entry(item).or_insert(0) += 1;
        }
        map
    }
}

impl<I: Iterator> IteratorExt for I {}

// Exercise 2: Newtype with Display
// Create a `Sorted<T>` newtype that wraps Vec<T> and keeps it sorted.
struct Sorted<T: Ord>(Vec<T>);

impl<T: Ord> Sorted<T> {
    fn new() -> Self { Sorted(Vec::new()) }

    fn insert(&mut self, item: T) {
        // TODO: Insert in sorted position
        let pos = self.0.binary_search(&item).unwrap_or_else(|p| p);
        self.0.insert(pos, item);
    }

    fn as_slice(&self) -> &[T] { &self.0 }
}

impl<T: Ord + fmt::Display> fmt::Display for Sorted<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.0.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{item}")?;
        }
        write!(f, "]")
    }
}

// Exercise 3: Type-state builder
// Create a URL builder that enforces scheme and host are set before build().
struct NoScheme;
struct HasScheme;
struct NoHost;
struct HasHost;

struct UrlBuilder<S, H> {
    scheme: Option<String>,
    host: Option<String>,
    path: Option<String>,
    _s: std::marker::PhantomData<(S, H)>,
}

impl UrlBuilder<NoScheme, NoHost> {
    fn new() -> Self {
        UrlBuilder {
            scheme: None, host: None, path: None,
            _s: std::marker::PhantomData,
        }
    }
}

impl<H> UrlBuilder<NoScheme, H> {
    fn scheme(self, s: &str) -> UrlBuilder<HasScheme, H> {
        UrlBuilder {
            scheme: Some(s.to_string()),
            host: self.host, path: self.path,
            _s: std::marker::PhantomData,
        }
    }
}

impl<S> UrlBuilder<S, NoHost> {
    fn host(self, h: &str) -> UrlBuilder<S, HasHost> {
        UrlBuilder {
            host: Some(h.to_string()),
            scheme: self.scheme, path: self.path,
            _s: std::marker::PhantomData,
        }
    }
}

impl UrlBuilder<HasScheme, HasHost> {
    fn path(mut self, p: &str) -> Self {
        self.path = Some(p.to_string());
        self
    }

    fn build(self) -> String {
        format!("{}://{}{}",
            self.scheme.unwrap(),
            self.host.unwrap(),
            self.path.unwrap_or_default())
    }
}

// Exercise 4: Trait object with clone
// TODO: Make Animal cloneable via trait objects
trait Animal: fmt::Debug {
    fn speak(&self) -> &str;
}

#[derive(Debug, Clone)]
struct Dog;
impl Animal for Dog { fn speak(&self) -> &str { "Woof!" } }

#[derive(Debug, Clone)]
struct Cat;
impl Animal for Cat { fn speak(&self) -> &str { "Meow!" } }

fn main() {
    // Test Exercise 1
    let words = vec!["hello", "world", "hello", "rust", "hello", "world"];
    let freq = words.into_iter().frequencies();
    assert_eq!(freq["hello"], 3);
    assert_eq!(freq["world"], 2);
    assert_eq!(freq["rust"], 1);
    println!("frequencies: {freq:?}");

    // Test Exercise 2
    let mut sorted = Sorted::new();
    sorted.insert(5);
    sorted.insert(2);
    sorted.insert(8);
    sorted.insert(1);
    sorted.insert(4);
    assert_eq!(sorted.as_slice(), &[1, 2, 4, 5, 8]);
    println!("Sorted: {sorted}");

    // Test Exercise 3
    let url = UrlBuilder::new()
        .scheme("https")
        .host("example.com")
        .path("/api/v1")
        .build();
    assert_eq!(url, "https://example.com/api/v1");
    println!("URL: {url}");

    // This should NOT compile (uncomment to verify):
    // let bad = UrlBuilder::new().host("example.com").build();

    // Test Exercise 4
    let animals: Vec<Box<dyn Animal>> = vec![Box::new(Dog), Box::new(Cat)];
    for a in &animals {
        println!("{:?} says {}", a, a.speak());
    }

    println!("\nAll exercises passed!");
}
