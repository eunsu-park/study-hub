# 08. 컬렉션(Collections)

**이전**: [열거형과 패턴 매칭](./07_Enums_and_Pattern_Matching.md) | **다음**: [에러 처리](./09_Error_Handling.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. `Vec<T>`, `String`, `HashMap<K, V>`를 생성, 수정, 순회할 수 있다
2. UTF-8 인코딩이 Rust의 문자열 인덱싱과 슬라이싱에 어떤 영향을 미치는지 설명할 수 있다
3. `Entry` API를 사용하여 해시맵에 조건부 삽입을 수행할 수 있다
4. 주어진 문제에 적합한 컬렉션 타입을 선택할 수 있다
5. 이터레이터 어댑터(`map`, `filter`, `collect`)를 연결하여 컬렉션을 변환할 수 있다

---

컬렉션(Collection)은 컴파일 타임에 크기가 고정된 배열이나 튜플과 달리, **힙(heap)**에 여러 값을 저장하는 데이터 구조입니다. Rust의 표준 라이브러리는 일상적으로 사용하게 될 세 가지 핵심 컬렉션을 제공합니다: 순서가 있는 시퀀스를 위한 `Vec<T>`, 소유된 텍스트를 위한 `String`, 키-값 조회를 위한 `HashMap<K, V>`. Rust 프로그래밍의 빵, 버터, 잼이라 할 수 있을 만큼, 사실상 모든 실질적인 프로그램은 이 중 최소 하나를 사용합니다.

## 목차
1. [Vec — 동적 배열](#1-vect--동적-배열)
2. [String — UTF-8 텍스트](#2-string--utf-8-텍스트)
3. [HashMap — 키-값 저장소](#3-hashmapk-v--키-값-저장소)
4. [BTreeMap vs HashMap](#4-btreemap-vs-hashmap)
5. [이터레이터 체이닝 기초](#5-이터레이터-체이닝-기초)
6. [VecDeque와 HashSet](#6-vecdeque와-hashset)
7. [연습 문제](#7-연습-문제)

---

## 1. Vec<T> — 동적 배열

`Vec<T>`는 힙에 저장되는 연속적이고 크기가 늘어나는 배열입니다. Python의 `list`나 C++의 `std::vector`를 사용해본 적 있다면, Rust의 `Vec`가 같은 역할을 한다고 생각하면 됩니다. 단, 소유권(ownership) 의미론이 내장되어 있습니다.

### 1.1 벡터 생성

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

### 1.2 용량(Capacity) vs 길이(Length)

차이를 이해하면 불필요한 메모리 할당을 방지할 수 있습니다:

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

### 1.3 요소 접근

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

### 1.4 순회(Iterating)

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

### 1.5 유용한 Vec 메서드

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

## 2. String — UTF-8 텍스트

Rust에는 두 가지 주요 문자열 타입이 있습니다:

| 타입 | 소유권 | 저장 위치 | 가변성 |
|------|--------|-----------|--------|
| `String` | 소유됨(Owned) | 힙(Heap) | 늘어날 수 있음 |
| `&str` | 빌림(Borrowed) | 어디서든 | 불변 슬라이스 |

`String`은 유효한 UTF-8을 보장하는 `Vec<u8>`이라고 생각하면 됩니다. `&str`는 UTF-8 바이트에 대한 `&[u8]` 뷰입니다.

### 2.1 문자열 생성

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

### 2.2 문자열 구성과 연결

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

### 2.3 UTF-8 인덱싱의 함정

Rust를 처음 접하는 사람에게 가장 놀라운 특징 중 하나입니다. `String`을 정수로 인덱싱할 수 **없습니다**:

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

### 2.4 문자열을 바라보는 세 가지 방법

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

### 2.5 유용한 String 메서드

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

## 3. HashMap<K, V> — 키-값 저장소

`HashMap`은 평균 O(1) 조회 성능으로 키-값 쌍을 저장합니다. Python의 `dict`나 JavaScript의 `Map`에 해당하는 Rust의 자료구조입니다.

### 3.1 생성과 삽입

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

### 3.2 값 접근

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

### 3.3 Entry API

`entry()` 메서드는 Rust에서 가장 우아한 API 중 하나입니다. "없으면 삽입, 있으면 수정"이라는 흔한 패턴을 이중 조회 없이 처리합니다:

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

### 3.4 단어 세기 — HashMap의 전형적인 패턴

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

### 3.5 HashMap의 소유권 규칙

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

Rust는 두 종류의 맵 타입을 제공합니다. 정렬이 필요한지에 따라 선택하세요:

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

## 5. 이터레이터 체이닝 기초

Rust 컬렉션은 이터레이터 어댑터와 결합할 때 진가를 발휘합니다. 이것은 맛보기이며, 레슨 12에서 이터레이터를 심층적으로 다룹니다.

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

## 6. VecDeque와 HashSet

초반에 알아두면 유용한 두 가지 컬렉션을 더 소개합니다:

### 6.1 VecDeque — 양방향 큐(Double-Ended Queue)

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

### 6.2 HashSet — 고유 값 집합

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

## 7. 연습 문제

### 문제 1: 빈도 계산기(Frequency Counter)
`char_frequency(s: &str) -> HashMap<char, usize>` 함수를 작성하세요. 문자열에서 각 문자(공백 제외)가 몇 번 등장하는지 셉니다. `"hello world"` 입력으로 테스트하여 `'l'`이 3번 등장하는지 확인하세요.

### 문제 2: 두 수의 합(Two Sum)
`Vec<i32>`와 목표값 `i32`가 주어질 때, 합이 목표값이 되는 두 숫자의 인덱스를 반환하세요. `HashMap`을 사용하여 O(n) 시간 복잡도를 달성하세요. 예: `[2, 7, 11, 15]`와 목표값 `9`가 주어지면 `(0, 1)`을 반환합니다.

### 문제 3: 순서를 유지하며 중복 제거
`unique_preserve_order(v: Vec<i32>) -> Vec<i32>` 함수를 작성하세요. 각 값의 첫 번째 등장 순서를 유지하면서 중복 값을 제거합니다. `HashSet`으로 이미 본 값을 추적하세요. 예: `[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]`은 `[3, 1, 4, 5, 9, 2, 6]`이 됩니다.

### 문제 4: 애너그램 그룹화(Group Anagrams)
`Vec<String>` 단어 목록을 받아 애너그램끼리 그룹화하는 함수를 작성하세요. 두 단어가 같은 문자를 임의의 순서로 포함하면 애너그램입니다(예: "eat", "tea", "ate"). `Vec<Vec<String>>`을 반환하세요. 힌트: 단어의 문자를 정렬하면 `HashMap`의 정규 키가 됩니다.

### 문제 5: 재고 시스템(Inventory System)
`HashMap<String, (u32, f64)>` (값은 `(수량, 단위가격)`)을 사용하는 간단한 재고 시스템을 만드세요. 다음 세 함수를 구현하세요:
- `add_item(inventory, name, quantity, price)` — 항목을 추가하거나 업데이트합니다
- `total_value(inventory) -> f64` — 모든 항목의 총 가치를 반환합니다
- `most_valuable(inventory) -> Option<String>` — `수량 * 가격`이 가장 높은 항목의 이름을 반환합니다

---

## 참고 자료

- [The Rust Programming Language, Ch. 8: Common Collections](https://doc.rust-lang.org/book/ch08-00-common-collections.html)
- [std::collections module documentation](https://doc.rust-lang.org/std/collections/index.html)
- [Rust by Example: Vectors](https://doc.rust-lang.org/rust-by-example/std/vec.html)
- [Rust by Example: Strings](https://doc.rust-lang.org/rust-by-example/std/str.html)
- [Rust by Example: HashMap](https://doc.rust-lang.org/rust-by-example/std/hash.html)

---

**이전**: [열거형과 패턴 매칭](./07_Enums_and_Pattern_Matching.md) | **다음**: [에러 처리](./09_Error_Handling.md)
