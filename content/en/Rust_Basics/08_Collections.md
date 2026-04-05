# 08. Collections

**Previous**: [Enums and Pattern Matching](./07_Enums_and_Pattern_Matching.md) | **Next**: [Error Handling](./09_Error_Handling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create, modify, and iterate over `Vec<T>`, `String`, and `HashMap<K, V>`
2. Explain how UTF-8 encoding affects string indexing and slicing in Rust
3. Use the `Entry` API to perform conditional insertion in hash maps
4. Choose the appropriate collection type for a given problem
5. Chain iterator adaptors (`map`, `filter`, `collect`) to transform collections

---

Collections are data structures that hold multiple values on the **heap**, unlike arrays and tuples which have a fixed size known at compile time. Rust's standard library provides three workhorses you will reach for daily: `Vec<T>` for ordered sequences, `String` for owned text, and `HashMap<K, V>` for key-value lookups. Think of them as the bread, butter, and jam of Rust programming — almost every non-trivial program uses at least one.

## Table of Contents
1. [Vec — The Growable Array](#1-vect--the-growable-array)
2. [String — UTF-8 Text](#2-string--utf-8-text)
3. [HashMap — Key-Value Storage](#3-hashmapk-v--key-value-storage)
4. [BTreeMap vs HashMap](#4-btreemap-vs-hashmap)
5. [Iterator Chaining Basics](#5-iterator-chaining-basics)
6. [VecDeque and HashSet](#6-vecdeque-and-hashset)
7. [Practice Problems](#7-practice-problems)

---

## 1. Vec<T> — The Growable Array

A `Vec<T>` is a contiguous, growable array stored on the heap. If you have used Python's `list` or C++'s `std::vector`, Rust's `Vec` fills the same role — but with ownership semantics baked in.

### 1.1 Creating Vectors

```rust
fn main() {
    // Method 1: Vec::new() — starts empty
    let mut numbers: Vec<i32> = Vec::new();
    numbers.push(1);
    numbers.push(2);
    numbers.push(3);

    // Method 2: vec! macro — the most common way
    let colors = vec!["red", "green", "blue"];

    // Method 3: Vec::with_capacity() — avoids reallocations
    // Use this when you know roughly how many elements you'll add
    let mut buffer = Vec::with_capacity(1000);
    buffer.push(42);

    println!("numbers: {:?}", numbers);
    println!("colors: {:?}", colors);
    println!("buffer len={}, capacity={}", buffer.len(), buffer.capacity());
    // Output: buffer len=1, capacity=1000
}
```

### 1.2 Capacity vs Length

Understanding the difference prevents unnecessary allocations:

```
Vec internals (on the heap):
                                          capacity = 8
           ┌───┬───┬───┬───┬───┬───┬───┬───┐
  data ──► │ 1 │ 2 │ 3 │ 4 │ 5 │   │   │   │
           └───┴───┴───┴───┴───┴───┴───┴───┘
                                  ▲
                            length = 5

  - length:   number of elements currently stored
  - capacity: total slots allocated (grows by doubling)
```

```rust
fn main() {
    let mut v = Vec::new();
    println!("len={}, cap={}", v.len(), v.capacity()); // 0, 0

    v.push(1);
    println!("len={}, cap={}", v.len(), v.capacity()); // 1, 4  (initial alloc)

    for i in 2..=5 {
        v.push(i);
    }
    println!("len={}, cap={}", v.len(), v.capacity()); // 5, 8  (doubled)

    // shrink_to_fit releases unused capacity
    v.shrink_to_fit();
    println!("len={}, cap={}", v.len(), v.capacity()); // 5, 5
}
```

### 1.3 Accessing Elements

```rust
fn main() {
    let v = vec![10, 20, 30, 40, 50];

    // Indexing — panics if out of bounds
    let third = v[2];
    println!("third = {}", third); // 30

    // .get() — returns Option<&T>, safe for uncertain indices
    match v.get(10) {
        Some(val) => println!("found {}", val),
        None => println!("index 10 is out of bounds"),
    }

    // Slicing — borrow a portion
    let middle = &v[1..4]; // [20, 30, 40]
    println!("middle: {:?}", middle);
}
```

### 1.4 Iterating

```rust
fn main() {
    let mut scores = vec![85, 92, 78, 96, 88];

    // Immutable iteration (borrows each element)
    for score in &scores {
        println!("score: {}", score);
    }

    // Mutable iteration (can modify in place)
    for score in &mut scores {
        *score += 5; // curve every score up by 5
    }

    // Consuming iteration (moves ownership — vec is gone after this)
    let total: i32 = scores.into_iter().sum();
    println!("total after curve: {}", total);
    // scores is no longer usable here
}
```

### 1.5 Useful Vec Methods

```rust
fn main() {
    let mut v = vec![3, 1, 4, 1, 5, 9, 2, 6];

    v.sort();                    // [1, 1, 2, 3, 4, 5, 6, 9]
    v.dedup();                   // [1, 2, 3, 4, 5, 6, 9]  — removes consecutive duplicates
    v.retain(|&x| x % 2 == 1);  // [1, 3, 5, 9]  — keep only odd numbers

    let last = v.pop();          // Some(9), v is now [1, 3, 5]
    println!("popped: {:?}", last);

    v.insert(1, 2);              // [1, 2, 3, 5]  — insert 2 at index 1
    let removed = v.remove(2);   // removes index 2 → returns 3, v is [1, 2, 5]
    println!("removed: {}", removed);

    println!("contains 5? {}", v.contains(&5)); // true
}
```

---

## 2. String — UTF-8 Text

Rust has two main string types:

| Type | Ownership | Storage | Mutability |
|------|-----------|---------|------------|
| `String` | Owned | Heap | Growable |
| `&str` | Borrowed | Anywhere | Immutable slice |

Think of `String` as `Vec<u8>` that guarantees valid UTF-8. And `&str` is a `&[u8]` view into UTF-8 bytes.

### 2.1 Creating Strings

```rust
fn main() {
    // From a string literal (&str → String)
    let s1 = String::from("hello");
    let s2 = "hello".to_string(); // equivalent

    // Empty string
    let s3 = String::new();

    // From formatted text
    let name = "Rust";
    let s4 = format!("Hello, {}!", name); // "Hello, Rust!"

    println!("{} | {} | '{}' | {}", s1, s2, s3, s4);
}
```

### 2.2 Building and Concatenating

```rust
fn main() {
    // push_str appends a &str slice
    let mut greeting = String::from("Hello");
    greeting.push_str(", world");

    // push appends a single char
    greeting.push('!');
    println!("{}", greeting); // "Hello, world!"

    // The + operator: consumes the left operand
    let hello = String::from("Hello");
    let world = String::from(" world");
    let combined = hello + &world; // hello is MOVED, world is borrowed
    // hello is no longer valid here
    println!("{}", combined);

    // format! is cleaner for complex concatenation (no moves)
    let first = String::from("tic");
    let second = String::from("tac");
    let third = String::from("toe");
    let game = format!("{}-{}-{}", first, second, third);
    println!("{}", game); // "tic-tac-toe"
    // first, second, third are still valid — format! only borrows
}
```

### 2.3 The UTF-8 Indexing Gotcha

This is one of Rust's most surprising features for newcomers. You **cannot** index a `String` by integer:

```rust
fn main() {
    let hello = String::from("Здравствуйте"); // Russian "Hello"

    // This WON'T COMPILE:
    // let h = hello[0];  // ERROR: String cannot be indexed by integer

    // Why? Because UTF-8 characters vary in byte length:
    //   'Z'  (Latin)    = 1 byte
    //   'д'  (Cyrillic) = 2 bytes
    //   '你' (CJK)      = 3 bytes
    //   '🦀' (Emoji)    = 4 bytes
}
```

```
UTF-8 encoding of "Здравствуйте" (12 characters, 24 bytes):

Byte index: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 ...
            ├──┤  ├──┤  ├──┤  ├──┤  ├──┤  ├──┤  ├──┤
Chars:       З     д     р     а     в     с     т    ...

Each Cyrillic character uses 2 bytes.
Indexing by byte position would split characters!
```

### 2.4 Three Ways to View a String

```rust
fn main() {
    let s = String::from("नमस्ते"); // Hindi "Namaste"

    // 1. Bytes — raw UTF-8 bytes (18 bytes for 6 characters)
    print!("bytes:  ");
    for b in s.bytes() {
        print!("{} ", b);
    }
    println!(); // 224 164 168 224 164 174 224 164 184 ...

    // 2. Chars — Unicode scalar values
    print!("chars:  ");
    for c in s.chars() {
        print!("'{}' ", c);
    }
    println!(); // 'न' 'म' 'स' '्' 'त' 'े'

    // 3. Byte slicing — you CAN slice, but must align to char boundaries
    let slice = &s[0..3]; // OK: first character is 3 bytes
    println!("slice: {}", slice); // "न"

    // &s[0..2] would PANIC at runtime — splits a character
}
```

### 2.5 Useful String Methods

```rust
fn main() {
    let s = String::from("  Hello, Rust World!  ");

    println!("{}", s.trim());                    // "Hello, Rust World!"
    println!("{}", s.trim().to_uppercase());     // "HELLO, RUST WORLD!"
    println!("{}", s.trim().to_lowercase());     // "hello, rust world!"
    println!("{}", s.trim().contains("Rust"));   // true
    println!("{}", s.trim().starts_with("Hello")); // true

    // Splitting
    let csv = "alice,bob,charlie";
    let names: Vec<&str> = csv.split(',').collect();
    println!("{:?}", names); // ["alice", "bob", "charlie"]

    // Replacing
    let fixed = "foo bar baz".replace("bar", "qux");
    println!("{}", fixed); // "foo qux baz"

    // Length: bytes vs characters
    let emoji = "Hello 🦀";
    println!("bytes: {}, chars: {}", emoji.len(), emoji.chars().count());
    // bytes: 10, chars: 7  (the crab emoji is 4 bytes)
}
```

---

## 3. HashMap<K, V> — Key-Value Storage

`HashMap` stores key-value pairs with O(1) average lookup. It is Rust's equivalent of Python's `dict` or JavaScript's `Map`.

### 3.1 Creating and Inserting

```rust
use std::collections::HashMap;

fn main() {
    // HashMap is not in the prelude — must import explicitly
    let mut scores: HashMap<String, i32> = HashMap::new();

    scores.insert(String::from("Alice"), 95);
    scores.insert(String::from("Bob"), 87);
    scores.insert(String::from("Charlie"), 92);

    println!("{:?}", scores);

    // From an iterator of tuples
    let teams = vec![
        ("Red", 3),
        ("Blue", 5),
        ("Green", 2),
    ];
    let standings: HashMap<&str, i32> = teams.into_iter().collect();
    println!("{:?}", standings);
}
```

### 3.2 Accessing Values

```rust
use std::collections::HashMap;

fn main() {
    let mut map = HashMap::new();
    map.insert("apple", 3);
    map.insert("banana", 7);

    // .get() returns Option<&V>
    if let Some(count) = map.get("apple") {
        println!("apple count: {}", count); // 3
    }

    // Indexing with [] — panics if key is missing
    // let x = map["cherry"]; // would panic!

    // Check existence
    println!("has banana? {}", map.contains_key("banana")); // true

    // Iterate over all key-value pairs
    for (fruit, count) in &map {
        println!("{}: {}", fruit, count);
    }
}
```

### 3.3 The Entry API

The `entry()` method is one of Rust's most elegant APIs. It handles the common pattern of "insert if absent, or modify if present" without double lookups:

```rust
use std::collections::HashMap;

fn main() {
    let mut scores: HashMap<&str, Vec<i32>> = HashMap::new();

    // or_insert: insert default if key is absent, return mutable ref either way
    scores.entry("Alice").or_insert(vec![]).push(95);
    scores.entry("Alice").or_insert(vec![]).push(87);
    scores.entry("Bob").or_insert(vec![]).push(92);

    println!("{:?}", scores);
    // {"Alice": [95, 87], "Bob": [92]}
}
```

### 3.4 Word Counting — A Classic HashMap Pattern

```rust
use std::collections::HashMap;

fn word_count(text: &str) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        // or_insert returns &mut V — we can dereference and increment
        let count = counts.entry(word).or_insert(0);
        *count += 1;
    }
    counts
}

fn main() {
    let text = "the quick brown fox jumps over the lazy fox";
    let counts = word_count(text);

    // Sort by count (descending) for display
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    for (word, count) in sorted {
        println!("{:>8}: {}", word, count);
    }
    // Output:
    //      the: 2
    //      fox: 2
    //    quick: 1
    //    brown: 1
    //    jumps: 1
    //     over: 1
    //     lazy: 1
}
```

### 3.5 Ownership Rules for HashMap

```rust
use std::collections::HashMap;

fn main() {
    let key = String::from("color");
    let value = String::from("blue");

    let mut map = HashMap::new();
    map.insert(key, value);
    // key and value are MOVED into the map — no longer valid here
    // println!("{}", key);  // ERROR: value moved

    // Types that implement Copy (like i32) are copied, not moved
    let mut nums = HashMap::new();
    let x = 42;
    nums.insert("answer", x);
    println!("x is still valid: {}", x); // OK: i32 is Copy
}
```

---

## 4. BTreeMap vs HashMap

Rust offers two map types. Choose based on whether you need ordering:

```
HashMap<K, V>                         BTreeMap<K, V>
┌────────────────────────┐            ┌────────────────────────┐
│ Hash table internally  │            │ B-Tree internally      │
│ K must impl: Hash + Eq │            │ K must impl: Ord       │
│ Lookup:  O(1) average  │            │ Lookup:  O(log n)      │
│ Insert:  O(1) average  │            │ Insert:  O(log n)      │
│ Ordered: NO            │            │ Ordered: YES (by key)  │
│ Use for: fast lookups  │            │ Use for: sorted output │
└────────────────────────┘            └────────────────────────┘
```

```rust
use std::collections::BTreeMap;

fn main() {
    let mut bt = BTreeMap::new();
    bt.insert("charlie", 3);
    bt.insert("alice", 1);
    bt.insert("bob", 2);

    // BTreeMap always iterates in key order
    for (name, id) in &bt {
        println!("{}: {}", name, id);
    }
    // alice: 1
    // bob: 2
    // charlie: 3

    // Range queries — only BTreeMap supports this
    for (name, id) in bt.range("alice"..="bob") {
        println!("in range: {} = {}", name, id);
    }
}
```

---

## 5. Iterator Chaining Basics

Rust collections shine when combined with iterator adaptors. This is a preview — Lesson 12 covers iterators in depth.

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // Chain: filter even numbers, square them, collect into a new Vec
    let even_squares: Vec<i32> = numbers
        .iter()              // create an iterator
        .filter(|&&n| n % 2 == 0)  // keep only even
        .map(|&n| n * n)    // square each
        .collect();          // gather results

    println!("{:?}", even_squares); // [4, 16, 36, 64, 100]

    // Iterators are lazy — nothing happens until collect() or another consumer
    // This makes chains efficient: no intermediate allocations

    // Sum of squares of odd numbers
    let sum: i32 = numbers
        .iter()
        .filter(|&&n| n % 2 == 1)
        .map(|&n| n * n)
        .sum(); // sum() is a consuming adaptor
    println!("sum of odd squares: {}", sum); // 1+9+25+49+81 = 165

    // Chaining with strings
    let sentence = "hello world from rust";
    let capitalized: String = sentence
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    first.to_uppercase().to_string() + chars.as_str()
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    println!("{}", capitalized); // "Hello World From Rust"
}
```

---

## 6. VecDeque and HashSet

Two more collections worth knowing early on:

### 6.1 VecDeque — Double-Ended Queue

```rust
use std::collections::VecDeque;

fn main() {
    let mut deque = VecDeque::new();

    // Efficient push/pop at BOTH ends (O(1) amortized)
    deque.push_back(1);
    deque.push_back(2);
    deque.push_front(0);
    println!("{:?}", deque); // [0, 1, 2]

    deque.pop_front(); // removes 0
    deque.pop_back();  // removes 2
    println!("{:?}", deque); // [1]

    // Use VecDeque when you need a queue (FIFO) or deque
    // Use Vec when you only push/pop from one end
}
```

### 6.2 HashSet — Unique Values

```rust
use std::collections::HashSet;

fn main() {
    let mut fruits: HashSet<&str> = HashSet::new();
    fruits.insert("apple");
    fruits.insert("banana");
    fruits.insert("apple"); // duplicate — silently ignored
    println!("count: {}", fruits.len()); // 2

    // Set operations
    let tropical: HashSet<&str> = ["banana", "mango", "papaya"].into();
    let temperate: HashSet<&str> = ["apple", "pear", "banana"].into();

    // Intersection: elements in both
    let both: HashSet<_> = tropical.intersection(&temperate).collect();
    println!("both: {:?}", both); // {"banana"}

    // Union: elements in either
    let all: HashSet<_> = tropical.union(&temperate).collect();
    println!("all: {:?}", all);

    // Difference: in tropical but not in temperate
    let only_tropical: HashSet<_> = tropical.difference(&temperate).collect();
    println!("only tropical: {:?}", only_tropical); // {"mango", "papaya"}
}
```

---

## 7. Practice Problems

### Problem 1: Frequency Counter
Write a function `char_frequency(s: &str) -> HashMap<char, usize>` that counts how many times each character (excluding whitespace) appears in a string. Test it with the input `"hello world"` and verify that `'l'` appears 3 times.

### Problem 2: Two Sum
Given a `Vec<i32>` and a target `i32`, return the indices of two numbers that add up to the target. Use a `HashMap` to achieve O(n) time complexity. For example, given `[2, 7, 11, 15]` and target `9`, return `(0, 1)`.

### Problem 3: Removing Duplicates While Preserving Order
Write a function `unique_preserve_order(v: Vec<i32>) -> Vec<i32>` that removes duplicate values from a vector while keeping the first occurrence of each value in its original position. Use a `HashSet` to track seen values. For example, `[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]` becomes `[3, 1, 4, 5, 9, 2, 6]`.

### Problem 4: Group Anagrams
Write a function that takes a `Vec<String>` of words and groups anagrams together. Two words are anagrams if they contain the same characters in any order (e.g., "eat", "tea", "ate"). Return a `Vec<Vec<String>>`. Hint: sorting the characters of a word produces a canonical key for the `HashMap`.

### Problem 5: Inventory System
Build a simple inventory system using a `HashMap<String, (u32, f64)>` where the value is `(quantity, price_per_unit)`. Implement three functions:
- `add_item(inventory, name, quantity, price)` — adds or updates an item
- `total_value(inventory) -> f64` — returns the total value of all items
- `most_valuable(inventory) -> Option<String>` — returns the name of the item with the highest `quantity * price`

---

## References

- [The Rust Programming Language, Ch. 8: Common Collections](https://doc.rust-lang.org/book/ch08-00-common-collections.html)
- [std::collections module documentation](https://doc.rust-lang.org/std/collections/index.html)
- [Rust by Example: Vectors](https://doc.rust-lang.org/rust-by-example/std/vec.html)
- [Rust by Example: Strings](https://doc.rust-lang.org/rust-by-example/std/str.html)
- [Rust by Example: HashMap](https://doc.rust-lang.org/rust-by-example/std/hash.html)

---

**Previous**: [Enums and Pattern Matching](./07_Enums_and_Pattern_Matching.md) | **Next**: [Error Handling](./09_Error_Handling.md)
