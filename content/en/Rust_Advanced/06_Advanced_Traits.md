# 22. Advanced Traits

**Previous**: [Procedural Macros](./05_Procedural_Macros.md) | **Next**: [Advanced Async](./07_Advanced_Async.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Choose between trait objects (`dyn Trait`) and generics (`impl Trait`) with informed trade-offs
2. Use Generic Associated Types (GATs) for lending iterators and async traits
3. Write blanket implementations and understand the orphan rule
4. Apply advanced associated type patterns including type-level state machines
5. Design trait hierarchies with supertraits, marker traits, and sealed traits

---

Lesson 10 introduced traits and generics. This lesson covers the advanced corners: trait objects vs monomorphization trade-offs, GATs (stabilized in Rust 1.65), the coherence and orphan rules, and patterns used in production Rust libraries.

## Table of Contents
1. [Static vs Dynamic Dispatch](#1-static-vs-dynamic-dispatch)
2. [Trait Objects in Depth](#2-trait-objects-in-depth)
3. [Object Safety](#3-object-safety)
4. [Associated Types vs Generic Parameters](#4-associated-types-vs-generic-parameters)
5. [Generic Associated Types (GATs)](#5-generic-associated-types-gats)
6. [Blanket Implementations](#6-blanket-implementations)
7. [Coherence and the Orphan Rule](#7-coherence-and-the-orphan-rule)
8. [Supertraits](#8-supertraits)
9. [Marker Traits](#9-marker-traits)
10. [Sealed Traits](#10-sealed-traits)
11. [Advanced Patterns](#11-advanced-patterns)
12. [Exercises](#12-exercises)

---

## 1. Static vs Dynamic Dispatch

Rust offers two dispatching mechanisms for trait-based polymorphism:

### Static Dispatch (Generics / `impl Trait`)

The compiler generates specialized code for each concrete type — **monomorphization**:

```rust
trait Drawable {
    fn draw(&self);
}

struct Circle { radius: f64 }
struct Square { side: f64 }

impl Drawable for Circle {
    fn draw(&self) { println!("Drawing circle r={}", self.radius); }
}
impl Drawable for Square {
    fn draw(&self) { println!("Drawing square s={}", self.side); }
}

// Static dispatch — compiler generates draw_shape::<Circle> and draw_shape::<Square>
fn draw_shape(shape: &impl Drawable) {
    shape.draw();
}

// Equivalent desugaring:
fn draw_shape_explicit<T: Drawable>(shape: &T) {
    shape.draw();
}

fn main() {
    draw_shape(&Circle { radius: 5.0 });
    draw_shape(&Square { side: 3.0 });
}
```

**Pros**: Zero-cost abstraction, inlining possible, no heap allocation.
**Cons**: Larger binary (code duplication), types must be known at compile time.

### Dynamic Dispatch (`dyn Trait`)

A vtable lookup at runtime — the compiler generates **one** copy of the function:

```rust
// Dynamic dispatch — single function handles any Drawable via vtable
fn draw_any(shape: &dyn Drawable) {
    shape.draw();  // vtable lookup at runtime
}

// Can store heterogeneous collections
fn draw_all(shapes: &[Box<dyn Drawable>]) {
    for shape in shapes {
        shape.draw();
    }
}

fn main() {
    let shapes: Vec<Box<dyn Drawable>> = vec![
        Box::new(Circle { radius: 5.0 }),
        Box::new(Square { side: 3.0 }),
    ];
    draw_all(&shapes);
}
```

**Pros**: Smaller binary, heterogeneous collections, runtime polymorphism.
**Cons**: vtable indirection cost, no inlining, requires heap allocation for owned values.

### Comparison Table

| Aspect | Static (`impl Trait`) | Dynamic (`dyn Trait`) |
|--------|----------------------|----------------------|
| Dispatch | Compile-time | Runtime (vtable) |
| Performance | Zero-cost | Small overhead |
| Binary size | Larger (monomorphized) | Smaller |
| Heterogeneous collection | No | Yes |
| Method inlining | Yes | No |
| Object safety required | No | Yes |

---

## 2. Trait Objects in Depth

A trait object (`dyn Trait`) is a **fat pointer** consisting of two pointers:

```
┌──────────────────────┐
│  data ptr  │ vtable ptr │
│  (8 bytes) │ (8 bytes)  │
└──────────────────────┘
```

The **data pointer** points to the concrete value. The **vtable pointer** points to a table of function pointers:

```
vtable for Circle's Drawable impl:
┌─────────────────────────────┐
│ drop_in_place: fn(*mut ())  │
│ size: usize                 │
│ align: usize                │
│ draw: fn(*const ())         │
└─────────────────────────────┘
```

### Trait Object Lifetimes

Trait objects have an implicit lifetime bound:

```rust
// Box<dyn Trait> is actually Box<dyn Trait + 'static>
// &'a dyn Trait is &'a (dyn Trait + 'a)

trait Logger {
    fn log(&self, msg: &str);
}

// Explicit lifetime on trait object
fn get_logger<'a>(loggers: &'a [Box<dyn Logger>]) -> &'a dyn Logger {
    &*loggers[0]
}

// Owned trait object — 'static by default
fn create_logger() -> Box<dyn Logger> {
    struct StdoutLogger;
    impl Logger for StdoutLogger {
        fn log(&self, msg: &str) { println!("{msg}"); }
    }
    Box::new(StdoutLogger)
}
```

### Multiple Trait Bounds on Trait Objects

```rust
use std::fmt::{Debug, Display};

// Cannot do: dyn Debug + Display (only one non-auto trait allowed)
// But you can create a supertrait:
trait DebugDisplay: Debug + Display {}
impl<T: Debug + Display> DebugDisplay for T {}

fn print_thing(thing: &dyn DebugDisplay) {
    println!("Debug: {:?}", thing);
    println!("Display: {}", thing);
}
```

---

## 3. Object Safety

Not all traits can be used as trait objects. A trait is **object-safe** if:

1. All methods have a receiver (`self`, `&self`, `&mut self`, `self: Box<Self>`, etc.)
2. No method returns `Self`
3. No method has generic type parameters
4. The trait does not require `Self: Sized`

```rust
// Object-safe
trait Draw {
    fn draw(&self);
    fn bounding_box(&self) -> (f64, f64, f64, f64);
}

// NOT object-safe — returns Self
trait Clonable {
    fn clone_self(&self) -> Self;  // Can't determine size at runtime
}

// NOT object-safe — generic method
trait Convertible {
    fn convert<T>(&self) -> T;  // vtable can't hold infinite specializations
}

// Workaround: make non-object-safe methods require Sized
trait MixedTrait {
    fn object_safe_method(&self);

    // This method is excluded from the trait object
    fn non_object_safe_method(&self) -> Self
    where
        Self: Sized;
}

// Now MixedTrait is object-safe
fn use_trait_object(obj: &dyn MixedTrait) {
    obj.object_safe_method();  // OK
    // obj.non_object_safe_method();  // ERROR: method requires Sized
}
```

### Clone for Trait Objects

```rust
// The standard Clone returns Self, so it's not object-safe.
// Workaround: define a clone-to-box method

trait ClonableAnimal: Animal {
    fn clone_box(&self) -> Box<dyn ClonableAnimal>;
}

impl<T: Animal + Clone + 'static> ClonableAnimal for T {
    fn clone_box(&self) -> Box<dyn ClonableAnimal> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ClonableAnimal> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

trait Animal {
    fn speak(&self) -> &str;
}

#[derive(Clone)]
struct Dog;
impl Animal for Dog {
    fn speak(&self) -> &str { "Woof!" }
}

fn main() {
    let a: Box<dyn ClonableAnimal> = Box::new(Dog);
    let b = a.clone();  // Works!
    println!("{}", b.speak());
}
```

---

## 4. Associated Types vs Generic Parameters

### When to Use Associated Types

Associated types enforce a **one-to-one** mapping: each implementing type chooses exactly one associated type:

```rust
// Iterator has an associated type — each iterator produces ONE type of item
trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

// A generic parameter would allow multiple implementations for the same type:
trait GenericIterator<Item> {
    fn next(&mut self) -> Option<Item>;
}

// With associated type: Vec<i32> -> Item = i32 (one choice)
// With generic: Vec<i32> could impl GenericIterator<i32> AND GenericIterator<String>
```

### Type-Level State Machines

Associated types enable compile-time state machines:

```rust
trait ConnectionState {}
struct Disconnected;
struct Connected;
struct Authenticated;

impl ConnectionState for Disconnected {}
impl ConnectionState for Connected {}
impl ConnectionState for Authenticated {}

struct Connection<S: ConnectionState> {
    host: String,
    _state: std::marker::PhantomData<S>,
}

impl Connection<Disconnected> {
    fn new(host: &str) -> Self {
        Connection {
            host: host.to_string(),
            _state: std::marker::PhantomData,
        }
    }

    fn connect(self) -> Connection<Connected> {
        println!("Connecting to {}", self.host);
        Connection {
            host: self.host,
            _state: std::marker::PhantomData,
        }
    }
}

impl Connection<Connected> {
    fn authenticate(self, _password: &str) -> Connection<Authenticated> {
        println!("Authenticating...");
        Connection {
            host: self.host,
            _state: std::marker::PhantomData,
        }
    }

    fn disconnect(self) -> Connection<Disconnected> {
        Connection {
            host: self.host,
            _state: std::marker::PhantomData,
        }
    }
}

impl Connection<Authenticated> {
    fn query(&self, sql: &str) -> String {
        format!("Running '{}' on {}", sql, self.host)
    }

    fn disconnect(self) -> Connection<Disconnected> {
        Connection {
            host: self.host,
            _state: std::marker::PhantomData,
        }
    }
}

fn main() {
    let conn = Connection::new("db.example.com");
    // conn.query("SELECT 1");  // ERROR: no method `query` on Connection<Disconnected>

    let conn = conn.connect();
    // conn.query("SELECT 1");  // ERROR: no method `query` on Connection<Connected>

    let conn = conn.authenticate("secret");
    let result = conn.query("SELECT 1");  // OK!
    println!("{result}");

    let _conn = conn.disconnect();
}
```

---

## 5. Generic Associated Types (GATs)

GATs (stabilized in Rust 1.65) allow associated types to have their own generic parameters:

```rust
trait LendingIterator {
    type Item<'a> where Self: 'a;

    fn next(&mut self) -> Option<Self::Item<'_>>;
}

// A windows iterator that lends slices
struct WindowsMut<'data, T> {
    data: &'data mut [T],
    pos: usize,
    size: usize,
}

impl<'data, T> LendingIterator for WindowsMut<'data, T> {
    type Item<'a> = &'a mut [T] where Self: 'a;

    fn next(&mut self) -> Option<Self::Item<'_>> {
        if self.pos + self.size > self.data.len() {
            return None;
        }
        let start = self.pos;
        self.pos += 1;
        // Reborrow to get the right lifetime
        Some(&mut self.data[start..start + self.size])
    }
}
```

### GATs for Async Traits (Pre-AFIT)

Before async fn in traits was stabilized, GATs were one workaround:

```rust
use std::future::Future;

trait AsyncProcessor {
    type ProcessFut<'a>: Future<Output = Vec<u8>> + 'a
    where
        Self: 'a;

    fn process<'a>(&'a self, data: &'a [u8]) -> Self::ProcessFut<'a>;
}

struct Compressor;

impl AsyncProcessor for Compressor {
    type ProcessFut<'a> = impl Future<Output = Vec<u8>> + 'a;

    fn process<'a>(&'a self, data: &'a [u8]) -> Self::ProcessFut<'a> {
        async move {
            // Simulate async compression
            data.to_vec()
        }
    }
}
```

### Collection Trait with GATs

```rust
trait Collection {
    type Item;
    type Iter<'a>: Iterator<Item = &'a Self::Item>
    where
        Self: 'a,
        Self::Item: 'a;

    fn iter(&self) -> Self::Iter<'_>;
    fn push(&mut self, item: Self::Item);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<T> Collection for Vec<T> {
    type Item = T;
    type Iter<'a> = std::slice::Iter<'a, T> where T: 'a;

    fn iter(&self) -> Self::Iter<'_> { self.as_slice().iter() }
    fn push(&mut self, item: T) { Vec::push(self, item); }
    fn len(&self) -> usize { Vec::len(self) }
}
```

---

## 6. Blanket Implementations

A blanket implementation implements a trait for all types satisfying certain bounds:

```rust
// From the standard library: every type that implements Display also
// implements ToString
impl<T: std::fmt::Display> ToString for T {
    fn to_string(&self) -> String {
        format!("{self}")
    }
}

// Your own blanket impl
trait Printable {
    fn print(&self);
}

impl<T: std::fmt::Debug> Printable for T {
    fn print(&self) {
        println!("{self:?}");
    }
}

fn main() {
    42.print();                // 42
    "hello".print();           // "hello"
    vec![1, 2, 3].print();    // [1, 2, 3]
}
```

### Extension Traits

Blanket impls power the **extension trait** pattern, which adds methods to foreign types:

```rust
trait IteratorExt: Iterator {
    fn take_every(self, n: usize) -> TakeEvery<Self>
    where
        Self: Sized,
    {
        TakeEvery { iter: self, n, count: 0 }
    }
}

// Blanket impl — every Iterator gets these methods
impl<I: Iterator> IteratorExt for I {}

struct TakeEvery<I> {
    iter: I,
    n: usize,
    count: usize,
}

impl<I: Iterator> Iterator for TakeEvery<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let item = self.iter.next()?;
            self.count += 1;
            if self.count % self.n == 0 {
                return Some(item);
            }
        }
    }
}

fn main() {
    let evens: Vec<_> = (1..=20).take_every(2).collect();
    println!("{evens:?}");  // [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
}
```

---

## 7. Coherence and the Orphan Rule

Rust's **coherence rules** ensure that for any type + trait combination, there is at most one implementation. The **orphan rule** is the key constraint:

> You can implement a trait for a type only if **either the trait or the type is local** to your crate.

```rust
// In YOUR crate:

// OK: your trait, foreign type
trait MyTrait {}
impl MyTrait for Vec<i32> {}

// OK: foreign trait, your type
struct MyType;
impl std::fmt::Display for MyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MyType")
    }
}

// ERROR: foreign trait, foreign type
// impl std::fmt::Display for Vec<i32> { ... }
```

### The Newtype Pattern (Workaround)

When you need to implement a foreign trait for a foreign type:

```rust
use std::fmt;

// Wrap the foreign type
struct PrettyVec<T>(Vec<T>);

impl<T: fmt::Display> fmt::Display for PrettyVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let items: Vec<String> = self.0.iter().map(|x| x.to_string()).collect();
        write!(f, "[{}]", items.join(", "))
    }
}

// Use Deref for transparent access
impl<T> std::ops::Deref for PrettyVec<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> { &self.0 }
}

fn main() {
    let v = PrettyVec(vec![1, 2, 3]);
    println!("{v}");     // [1, 2, 3]   — uses our Display
    println!("{}", v.len());  // 3     — uses Deref to Vec
}
```

---

## 8. Supertraits

A supertrait is a trait that requires another trait to be implemented:

```rust
use std::fmt;

// Display is a supertrait of Describable
trait Describable: fmt::Display + fmt::Debug {
    fn description(&self) -> String {
        format!("Display: {} | Debug: {:?}", self, self)
    }
}

#[derive(Debug)]
struct Product {
    name: String,
    price: f64,
}

impl fmt::Display for Product {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (${:.2})", self.name, self.price)
    }
}

impl Describable for Product {}

fn print_description(item: &dyn Describable) {
    // Can use Display and Debug because they're supertraits
    println!("Description: {}", item.description());
    println!("Display: {item}");
    println!("Debug: {item:?}");
}

fn main() {
    let p = Product { name: "Widget".into(), price: 9.99 };
    print_description(&p);
}
```

---

## 9. Marker Traits

Marker traits carry no methods — they convey semantic information to the compiler or other code:

```rust
// Standard library markers:
// Send — safe to transfer between threads
// Sync — safe to share references between threads
// Sized — has a known size at compile time
// Unpin — safe to move after being pinned

// Custom marker trait
trait ThreadSafeCache: Send + Sync {}

trait Immutable {}  // Marker: type promises not to mutate

// Use marker trait as a bound
fn process<T: Immutable>(data: &T) {
    // We trust that T doesn't do interior mutation
}

// Negative impls (unstable, but conceptually):
// impl !Send for MyRawPointerWrapper {}

// Auto traits — automatically implemented for types whose fields all implement it
// Send and Sync are auto traits
```

### Using PhantomData with Marker Traits

```rust
use std::marker::PhantomData;

struct Readonly;
struct Writable;

struct Handle<Mode> {
    id: u64,
    _mode: PhantomData<Mode>,
}

impl Handle<Readonly> {
    fn read(&self) -> String {
        format!("Reading from handle {}", self.id)
    }
}

impl Handle<Writable> {
    fn read(&self) -> String {
        format!("Reading from handle {}", self.id)
    }
    fn write(&self, data: &str) {
        println!("Writing '{data}' to handle {}", self.id);
    }
}

fn open_readonly(id: u64) -> Handle<Readonly> {
    Handle { id, _mode: PhantomData }
}

fn open_writable(id: u64) -> Handle<Writable> {
    Handle { id, _mode: PhantomData }
}

fn main() {
    let r = open_readonly(1);
    println!("{}", r.read());
    // r.write("data");  // ERROR: no method `write` on Handle<Readonly>

    let w = open_writable(2);
    println!("{}", w.read());
    w.write("some data");  // OK
}
```

---

## 10. Sealed Traits

A sealed trait can be implemented only within the crate that defines it. This allows you to add methods later without breaking downstream code:

```rust
mod private {
    pub trait Sealed {}
}

// Public trait with a private supertrait
pub trait DatabaseDriver: private::Sealed {
    fn connect(&self, url: &str) -> String;
    fn query(&self, sql: &str) -> Vec<String>;
}

// Only types in this crate can implement Sealed, and thus DatabaseDriver
pub struct PostgresDriver;
impl private::Sealed for PostgresDriver {}
impl DatabaseDriver for PostgresDriver {
    fn connect(&self, url: &str) -> String {
        format!("Connected to Postgres at {url}")
    }
    fn query(&self, sql: &str) -> Vec<String> {
        vec![format!("Postgres result for: {sql}")]
    }
}

pub struct SqliteDriver;
impl private::Sealed for SqliteDriver {}
impl DatabaseDriver for SqliteDriver {
    fn connect(&self, url: &str) -> String {
        format!("Connected to SQLite at {url}")
    }
    fn query(&self, sql: &str) -> Vec<String> {
        vec![format!("SQLite result for: {sql}")]
    }
}

// External crates CANNOT do:
// impl private::Sealed for MyDriver {}  // Error: Sealed is in a private module
// impl DatabaseDriver for MyDriver {}   // Error: missing Sealed bound
```

---

## 11. Advanced Patterns

### Pattern: Type-Level Numbers

```rust
trait TypeNum {
    const VALUE: usize;
}

struct Zero;
struct Succ<N>(std::marker::PhantomData<N>);

impl TypeNum for Zero {
    const VALUE: usize = 0;
}

impl<N: TypeNum> TypeNum for Succ<N> {
    const VALUE: usize = N::VALUE + 1;
}

type One = Succ<Zero>;
type Two = Succ<One>;
type Three = Succ<Two>;

fn print_type_num<N: TypeNum>() {
    println!("Type-level number: {}", N::VALUE);
}

fn main() {
    print_type_num::<Zero>();   // 0
    print_type_num::<One>();    // 1
    print_type_num::<Three>();  // 3
}
```

### Pattern: Strategy with Traits

```rust
trait SortStrategy {
    fn sort<T: Ord>(data: &mut [T]);
}

struct BubbleSort;
struct QuickSort;

impl SortStrategy for BubbleSort {
    fn sort<T: Ord>(data: &mut [T]) {
        let n = data.len();
        for i in 0..n {
            for j in 0..n - 1 - i {
                if data[j] > data[j + 1] {
                    data.swap(j, j + 1);
                }
            }
        }
    }
}

impl SortStrategy for QuickSort {
    fn sort<T: Ord>(data: &mut [T]) {
        data.sort();  // Use standard library quicksort
    }
}

struct Sorter<S: SortStrategy> {
    _strategy: std::marker::PhantomData<S>,
}

impl<S: SortStrategy> Sorter<S> {
    fn new() -> Self {
        Sorter { _strategy: std::marker::PhantomData }
    }

    fn sort<T: Ord>(&self, data: &mut [T]) {
        S::sort(data);
    }
}

fn main() {
    let mut data = vec![5, 3, 8, 1, 9, 2];

    let sorter = Sorter::<BubbleSort>::new();
    sorter.sort(&mut data);
    println!("Bubble sorted: {data:?}");

    let mut data = vec![5, 3, 8, 1, 9, 2];
    let sorter = Sorter::<QuickSort>::new();
    sorter.sort(&mut data);
    println!("Quick sorted: {data:?}");
}
```

---

## 12. Exercises

1. **Plugin system**: Design a trait-based plugin system where plugins can be loaded dynamically (`Box<dyn Plugin>`) and each plugin declares its name, version, and an `execute` method.

2. **GAT collection**: Implement the `Collection` GAT trait for a `LinkedList` type (you can use `std::collections::LinkedList`).

3. **Newtype with full delegation**: Create a `SortedVec<T>` newtype around `Vec<T>` that maintains sorted order. Implement `Deref`, `Display`, `IntoIterator`, and `FromIterator`.

4. **Sealed trait hierarchy**: Design a sealed `Codec` trait with `encode`/`decode` methods. Implement it for `Json`, `Toml`, and `Yaml` types. Ensure external crates cannot add codecs.

5. **Type-state builder**: Create a `RequestBuilder` that uses the typestate pattern to ensure `url`, `method`, and `body` are set before `send()` can be called (compile-time enforcement).

---

## References

- [The Rust Reference: Trait Objects](https://doc.rust-lang.org/reference/types/trait-object.html)
- [RFC 1598: GATs](https://rust-lang.github.io/rfcs/1598-generic_associated_types.html)
- [Rust Blog: GATs Stabilization](https://blog.rust-lang.org/2022/10/28/gats-stabilization.html)
- [Coherence and Orphan Rules](https://doc.rust-lang.org/reference/items/implementations.html#orphan-rules)

---

**Previous**: [Procedural Macros](./05_Procedural_Macros.md) | **Next**: [Advanced Async](./07_Advanced_Async.md)
