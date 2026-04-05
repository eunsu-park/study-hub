# 13. Smart Pointers

**Previous**: [Closures and Iterators](./12_Closures_and_Iterators.md) | **Next**: [Concurrency](./14_Concurrency.md)

**Difficulty**: ⭐⭐⭐

## Learning Objectives

- Explain the difference between smart pointers and regular references, and when heap allocation is necessary
- Use `Box<T>` for heap allocation, recursive data structures, and trait objects
- Implement the `Deref` and `Drop` traits to customize pointer behavior and resource cleanup
- Apply `Rc<T>` and `RefCell<T>` to model shared and interior-mutable ownership in single-threaded code
- Detect and break reference cycles using `Weak<T>`

## Table of Contents

1. [Smart Pointers vs References](#1-smart-pointers-vs-references)
2. [Box: Heap Allocation](#2-boxt-heap-allocation)
3. [The Deref Trait](#3-the-deref-trait)
4. [The Drop Trait](#4-the-drop-trait)
5. [Rc: Reference Counting](#5-rct-reference-counting)
6. [RefCell: Interior Mutability](#6-refcellt-interior-mutability)
7. [Rc RefCell Pattern](#7-rcrefcellt-pattern)
8. [Arc: Atomic Reference Counting](#8-arct-atomic-reference-counting)
9. [Weak: Breaking Reference Cycles](#9-weakt-breaking-reference-cycles)
10. [Smart Pointer Comparison Table](#10-smart-pointer-comparison-table)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Smart Pointers vs References

A **reference** (`&T` or `&mut T`) is a lightweight pointer that borrows data without owning it. A **smart pointer** is a struct that acts like a pointer but also **owns** the data it points to and provides additional capabilities (automatic cleanup, reference counting, interior mutability, etc.).

Think of the difference like this: a reference is like borrowing a friend's book — you can read it but must return it. A smart pointer is like buying your own copy — you own it, and when you are done, you decide what happens to it.

```
Reference (&T)              Smart Pointer (e.g., Box<T>)
──────────────              ────────────────────────────
Borrows data                Owns data
No metadata                 May carry metadata (ref count, etc.)
No destructor               Implements Drop for cleanup
Always valid (borrow rules) Manages its own validity
Stack only (pointer itself)  Data lives on the heap
```

Smart pointers in Rust implement two key traits:
- **`Deref`** — allows the smart pointer to behave like a reference (use `*` operator)
- **`Drop`** — defines what happens when the smart pointer goes out of scope

---

## 2. Box<T>: Heap Allocation

`Box<T>` is the simplest smart pointer. It allocates data on the **heap** and stores a pointer on the stack. When the `Box` goes out of scope, both the heap data and the pointer are freed.

```
Stack                  Heap
┌──────────┐          ┌──────────┐
│ Box<i32> │ ───────► │  42      │
│ (8 bytes)│          │ (4 bytes)│
└──────────┘          └──────────┘
```

### Basic Usage

```rust
fn main() {
    // Allocate an integer on the heap
    let boxed = Box::new(42);
    println!("boxed = {boxed}"); // Deref makes this work transparently

    // Box is useful when you need a known size for an unsized type
    let boxed_slice: Box<[i32]> = vec![1, 2, 3].into_boxed_slice();
    println!("slice len = {}", boxed_slice.len());
}
```

### Recursive Types

Without `Box`, recursive types have infinite size. `Box` provides indirection with a known pointer size:

```rust
// A linked list: each node holds a value and a pointer to the next node
#[derive(Debug)]
enum List {
    Cons(i32, Box<List>),  // Box provides a fixed-size pointer
    Nil,
}

use List::{Cons, Nil};

fn main() {
    // Build list: 1 -> 2 -> 3 -> Nil
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
    println!("{list:?}");
}
```

Without `Box`, the compiler would complain:

```
error: recursive type `List` has infinite size
  --> help: insert some indirection (e.g., a `Box`) to break the cycle
```

### Trait Objects

`Box<dyn Trait>` enables dynamic dispatch — storing different concrete types behind a common interface:

```rust
trait Drawable {
    fn draw(&self);
}

struct Circle { radius: f64 }
struct Square { side: f64 }

impl Drawable for Circle {
    fn draw(&self) { println!("Drawing circle with r={}", self.radius); }
}
impl Drawable for Square {
    fn draw(&self) { println!("Drawing square with s={}", self.side); }
}

fn main() {
    // Heterogeneous collection: different types behind a common trait
    let shapes: Vec<Box<dyn Drawable>> = vec![
        Box::new(Circle { radius: 3.0 }),
        Box::new(Square { side: 4.0 }),
        Box::new(Circle { radius: 1.5 }),
    ];

    for shape in &shapes {
        shape.draw(); // dynamic dispatch via vtable
    }
}
```

---

## 3. The Deref Trait

The `Deref` trait lets you customize the behavior of the dereference operator `*`. This enables **deref coercion**, where the compiler automatically converts a reference to a smart pointer into a reference to the inner data.

```rust
use std::ops::Deref;

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(val: T) -> MyBox<T> {
        MyBox(val)
    }
}

// Implementing Deref allows *MyBox<T> to yield &T
impl<T> Deref for MyBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0 // return a reference to the inner value
    }
}

fn greet(name: &str) {
    println!("Hello, {name}!");
}

fn main() {
    let boxed_name = MyBox::new(String::from("Rust"));

    // Deref coercion chain:
    //   &MyBox<String> → &String → &str
    // The compiler inserts these conversions automatically
    greet(&boxed_name);

    // Explicit dereference (rarely needed thanks to coercion)
    let inner: &String = &*boxed_name;
    println!("inner = {inner}");
}
```

Deref coercion follows these rules:
- `&T` to `&U` when `T: Deref<Target = U>` (immutable)
- `&mut T` to `&mut U` when `T: DerefMut<Target = U>` (mutable)
- `&mut T` to `&U` when `T: Deref<Target = U>` (mutable to immutable is always safe)

---

## 4. The Drop Trait

The `Drop` trait defines custom cleanup logic that runs when a value goes out of scope. This is Rust's equivalent of a destructor — it handles resource release (closing files, freeing memory, releasing locks).

```rust
struct DatabaseConnection {
    name: String,
}

impl DatabaseConnection {
    fn new(name: &str) -> Self {
        println!("[{name}] Connection opened");
        DatabaseConnection { name: name.to_string() }
    }
}

impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        // Automatic cleanup when the connection goes out of scope
        println!("[{}] Connection closed (cleaned up)", self.name);
    }
}

fn main() {
    let _conn1 = DatabaseConnection::new("primary");
    let _conn2 = DatabaseConnection::new("replica");

    println!("Doing work...");

    // Drop runs in reverse declaration order when main() ends:
    // [replica] Connection closed (cleaned up)
    // [primary] Connection closed (cleaned up)
}
```

### Early Drop with `std::mem::drop`

You cannot call `.drop()` directly (the compiler forbids it to prevent double-free). Instead, use `std::mem::drop()`:

```rust
fn main() {
    let conn = DatabaseConnection::new("temp");
    println!("Using connection...");

    drop(conn); // explicitly drop early — triggers Drop::drop()
    // `conn` is no longer valid here

    println!("Connection already cleaned up, continuing...");
}

// (DatabaseConnection definition from above)
struct DatabaseConnection { name: String }
impl DatabaseConnection {
    fn new(name: &str) -> Self {
        println!("[{name}] Connection opened");
        DatabaseConnection { name: name.to_string() }
    }
}
impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        println!("[{}] Connection closed", self.name);
    }
}
```

---

## 5. Rc<T>: Reference Counting

Sometimes a value needs **multiple owners**. For example, in a graph, multiple edges may point to the same node. `Rc<T>` (Reference Counted) tracks how many `Rc` pointers refer to the data and frees it when the count reaches zero.

```
   owner_a ──────►┌──────────────────┐
                  │ Rc<String>       │
   owner_b ──────►│ ref_count: 3     │
                  │ data: "shared"   │
   owner_c ──────►└──────────────────┘
```

```rust
use std::rc::Rc;

fn main() {
    let original = Rc::new(String::from("shared data"));
    println!("ref count after creation: {}", Rc::strong_count(&original)); // 1

    let clone_a = Rc::clone(&original); // increment ref count, NOT deep copy
    println!("ref count after clone_a: {}", Rc::strong_count(&original)); // 2

    {
        let clone_b = Rc::clone(&original);
        println!("ref count with clone_b: {}", Rc::strong_count(&original)); // 3
    } // clone_b dropped here

    println!("ref count after clone_b dropped: {}", Rc::strong_count(&original)); // 2
    println!("data: {original}");
}
```

**Important limitations** of `Rc<T>`:
- **Single-threaded only** — not safe to send across threads (use `Arc<T>` instead)
- **Immutable data only** — `Rc<T>` gives you shared `&T` references, not `&mut T`
- To get mutability with `Rc`, combine it with `RefCell<T>` (see below)

---

## 6. RefCell<T>: Interior Mutability

Normally, Rust enforces borrowing rules at **compile time**: you can have either one `&mut T` or many `&T`, but not both. `RefCell<T>` moves these checks to **runtime**, allowing you to mutate data even when you only have an immutable reference.

```
Compile-time borrowing (&T / &mut T):     Runtime borrowing (RefCell<T>):
──────────────────────────────────────    ──────────────────────────────────
Errors caught at compilation              Errors cause panic at runtime
Zero runtime cost                         Small runtime overhead for tracking
Restrictive but safe                      Flexible but risks runtime panics
```

```rust
use std::cell::RefCell;

fn main() {
    let data = RefCell::new(vec![1, 2, 3]);

    // borrow() returns Ref<T> — an immutable borrow tracked at runtime
    println!("data = {:?}", data.borrow());

    // borrow_mut() returns RefMut<T> — a mutable borrow tracked at runtime
    data.borrow_mut().push(4);
    println!("after push: {:?}", data.borrow());

    // Runtime panic if borrowing rules are violated:
    // let r1 = data.borrow();
    // let r2 = data.borrow_mut(); // PANIC: already borrowed immutably
}
```

A practical use case is the **mock object** pattern in testing, where you want to record method calls on an otherwise immutable reference:

```rust
use std::cell::RefCell;

struct MockLogger {
    messages: RefCell<Vec<String>>,
}

impl MockLogger {
    fn new() -> Self {
        MockLogger { messages: RefCell::new(Vec::new()) }
    }

    // Note: &self (immutable), but we can still record messages
    fn log(&self, msg: &str) {
        self.messages.borrow_mut().push(msg.to_string());
    }

    fn get_messages(&self) -> Vec<String> {
        self.messages.borrow().clone()
    }
}

fn main() {
    let logger = MockLogger::new();

    // log() takes &self, but internally mutates via RefCell
    logger.log("Starting process");
    logger.log("Process complete");

    println!("Logged: {:?}", logger.get_messages());
}
```

---

## 7. Rc<RefCell<T>> Pattern

Combining `Rc` and `RefCell` gives you **shared mutable ownership**: multiple owners that can all mutate the inner data.

```
   owner_a ──────►┌──────────────────────────┐
                  │ Rc<RefCell<Vec<i32>>>     │
   owner_b ──────►│ ref_count: 2             │
                  │ RefCell { data: [1,2,3] }│
                  └──────────────────────────┘
                  Both owners can borrow_mut() the Vec
```

```rust
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug)]
struct Node {
    value: i32,
    children: Vec<Rc<RefCell<Node>>>,
}

impl Node {
    fn new(value: i32) -> Rc<RefCell<Node>> {
        Rc::new(RefCell::new(Node {
            value,
            children: Vec::new(),
        }))
    }
}

fn main() {
    let root = Node::new(1);
    let child_a = Node::new(2);
    let child_b = Node::new(3);

    // Add children to root — root borrows mutably via RefCell
    root.borrow_mut().children.push(Rc::clone(&child_a));
    root.borrow_mut().children.push(Rc::clone(&child_b));

    // child_a is shared: root owns it, and we can still access it directly
    child_a.borrow_mut().value = 20;

    // Print the tree
    let root_ref = root.borrow();
    println!("Root: {}", root_ref.value);
    for child in &root_ref.children {
        println!("  Child: {}", child.borrow().value);
    }
}
```

**When to use this pattern**: tree structures, graphs, observer patterns — anywhere you need multiple owners with mutation. But prefer simpler alternatives (indices into a `Vec`, `HashMap`) when possible, as `Rc<RefCell<T>>` trades compile-time safety for runtime checks.

---

## 8. Arc<T>: Atomic Reference Counting

`Arc<T>` (Atomic Reference Counted) is the thread-safe version of `Rc<T>`. It uses atomic operations to update the reference count, making it safe to share across threads.

```rust
use std::sync::Arc;
use std::thread;

fn main() {
    let data = Arc::new(vec![1, 2, 3, 4, 5]);

    let mut handles = vec![];

    for i in 0..3 {
        let data_clone = Arc::clone(&data); // atomic ref count increment
        let handle = thread::spawn(move || {
            // Each thread has its own Arc pointing to the same data
            let sum: i32 = data_clone.iter().sum();
            println!("Thread {i}: sum = {sum}");
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final ref count: {}", Arc::strong_count(&data)); // 1
}
```

`Arc` vs `Rc`:

| Feature | `Rc<T>` | `Arc<T>` |
|---------|---------|----------|
| Thread-safe | No (`!Send`, `!Sync`) | Yes |
| Performance | Faster (no atomic ops) | Slight overhead (atomic ops) |
| Use case | Single-threaded sharing | Multi-threaded sharing |

For mutable shared state across threads, combine with `Mutex<T>`: `Arc<Mutex<T>>` (covered in detail in Lesson 14).

---

## 9. Weak<T>: Breaking Reference Cycles

`Rc` and `Arc` can create **reference cycles** — two values that point to each other, preventing the ref count from ever reaching zero. This causes memory leaks. `Weak<T>` solves this by providing a non-owning reference that does not increment the strong count.

```
Strong cycle (memory leak):          With Weak (no leak):
┌────────┐     ┌────────┐          ┌────────┐      ┌────────┐
│ Node A │────►│ Node B │          │ Node A │─────►│ Node B │
│ rc: 2  │◄────│ rc: 2  │          │ rc: 1  │◄─ ─ ─│ rc: 1  │
└────────┘     └────────┘          └────────┘weak  └────────┘
Both rc never reach 0!              Weak ref doesn't prevent drop
```

```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;

#[derive(Debug)]
struct Person {
    name: String,
    // A person's best friend — Weak avoids a reference cycle
    best_friend: RefCell<Option<Weak<Person>>>,
}

impl Person {
    fn new(name: &str) -> Rc<Person> {
        Rc::new(Person {
            name: name.to_string(),
            best_friend: RefCell::new(None),
        })
    }
}

impl Drop for Person {
    fn drop(&mut self) {
        println!("{} is being dropped", self.name);
    }
}

fn main() {
    let alice = Person::new("Alice");
    let bob = Person::new("Bob");

    // Set up mutual friendship using Weak references
    *alice.best_friend.borrow_mut() = Some(Rc::downgrade(&bob));
    *bob.best_friend.borrow_mut() = Some(Rc::downgrade(&alice));

    // Access via Weak::upgrade() — returns Option<Rc<T>>
    if let Some(friend) = alice.best_friend.borrow().as_ref().unwrap().upgrade() {
        println!("Alice's best friend is {}", friend.name);
    }

    println!("Alice strong={}, weak={}", Rc::strong_count(&alice), Rc::weak_count(&alice));
    println!("Bob   strong={}, weak={}", Rc::strong_count(&bob), Rc::weak_count(&bob));

    // Both Alice and Bob will be dropped properly — no memory leak
}
```

**Rule of thumb**: use `Weak` for "back-pointers" — parent references in trees, inverse relationships in graphs, caches where entries should not prevent deallocation.

---

## 10. Smart Pointer Comparison Table

| Type | Ownership | Mutability | Thread-safe | Use Case |
|------|-----------|------------|-------------|----------|
| `Box<T>` | Single owner | If owner is mutable | Yes (`Send + Sync` if `T` is) | Heap allocation, recursive types, trait objects |
| `Rc<T>` | Multiple owners | Immutable | No | Shared ownership, single-threaded |
| `Arc<T>` | Multiple owners | Immutable | Yes | Shared ownership, multi-threaded |
| `RefCell<T>` | Single owner | Interior mutability | No | Runtime borrow checking |
| `Mutex<T>` | Single owner | Interior mutability | Yes | Thread-safe interior mutability |
| `Rc<RefCell<T>>` | Multiple + mutable | Interior mutability | No | Shared mutable state, single-threaded |
| `Arc<Mutex<T>>` | Multiple + mutable | Interior mutability | Yes | Shared mutable state, multi-threaded |
| `Weak<T>` | Non-owning | Immutable (via upgrade) | Matches parent | Breaking reference cycles |

Decision flowchart:

```
Need heap allocation?
├── Yes, single owner → Box<T>
├── Yes, multiple owners, single thread → Rc<T>
│   └── Need mutation? → Rc<RefCell<T>>
├── Yes, multiple owners, multi-thread → Arc<T>
│   └── Need mutation? → Arc<Mutex<T>>
└── No → use stack allocation (default)
```

---

## 11. Practice Problems

### Problem 1: Expression Tree Evaluator

Build a binary expression tree using `Box<T>`. Define an `Expr` enum with variants `Num(f64)`, `Add(Box<Expr>, Box<Expr>)`, `Mul(Box<Expr>, Box<Expr>)`, and `Neg(Box<Expr>)`. Implement an `eval()` method that recursively evaluates the expression. Test with `(3 + 4) * -(2 + 5)`.

### Problem 2: Reference-Counted Document Model

Create a `Document` struct with `Rc<String>` content. Implement a `Snapshot` system: calling `snapshot()` returns a lightweight copy (via `Rc::clone`) that shares the content. When the document is modified, it should allocate a new `String` (copy-on-write semantics). Verify that old snapshots retain the original content.

### Problem 3: Doubly-Linked List

Implement a doubly-linked list using `Rc`, `Weak`, and `RefCell`. Each node should have a `next: Option<Rc<RefCell<Node>>>` and `prev: Option<Weak<RefCell<Node>>>`. Implement `push_back`, `push_front`, and `display_forward` methods. Verify that all nodes are properly dropped (no memory leaks).

### Problem 4: Custom Smart Pointer with Logging

Create a `TrackedBox<T>` smart pointer that logs every dereference and drop. Implement `Deref`, `DerefMut`, and `Drop`. Use a `static AtomicUsize` counter to track total allocations and deallocations. Print a summary showing whether all allocations were freed.

### Problem 5: Observer Pattern

Implement the Observer pattern using `Rc<RefCell<T>>` and `Weak<RefCell<T>>`. A `Subject` holds a list of `Weak` references to observers. Observers register themselves and receive notifications. Demonstrate that dropping an observer does not prevent the subject from functioning (the `Weak` reference simply fails to upgrade).

---

## 12. References

- [The Rust Programming Language, Ch. 15: Smart Pointers](https://doc.rust-lang.org/book/ch15-00-smart-pointers.html)
- [Rust by Example: Box, Rc, Arc](https://doc.rust-lang.org/rust-by-example/std/box.html)
- [std::rc::Rc Documentation](https://doc.rust-lang.org/std/rc/struct.Rc.html)
- [std::cell::RefCell Documentation](https://doc.rust-lang.org/std/cell/struct.RefCell.html)
- [std::sync::Arc Documentation](https://doc.rust-lang.org/std/sync/struct.Arc.html)

---

**Previous**: [Closures and Iterators](./12_Closures_and_Iterators.md) | **Next**: [Concurrency](./14_Concurrency.md)
