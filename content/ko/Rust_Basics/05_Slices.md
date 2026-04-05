# 05. 슬라이스(Slices)

**이전**: [빌림과 참조(Borrowing and References)](./04_Borrowing_and_References.md) | **다음**: [구조체와 메서드(Structs and Methods)](./06_Structs_and_Methods.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 소유된 데이터로부터 문자열 슬라이스(`&str`)와 배열 슬라이스(`&[T]`)를 생성한다
2. `String`과 `&str`의 차이를 설명하고 각각을 언제 사용할지 이해한다
3. 유연성을 극대화하기 위해 `&str`을 받는 함수를 작성한다
4. 검색, 분할, 반복을 위한 슬라이스 메서드를 사용한다

---

슬라이스(Slice)는 컬렉션 전체가 아닌 연속된 요소들의 시퀀스에 대한 참조입니다. 데이터를 복사하지 않고 일부를 다룰 수 있으며, Rust에서 문자열과 배열을 전달하는 관용적인 방식입니다.

## 목차
1. [문자열 슬라이스](#1-문자열-슬라이스)
2. [String vs &str](#2-string-vs-str)
3. [배열과 Vec 슬라이스](#3-배열과-vec-슬라이스)
4. [슬라이스 메서드](#4-슬라이스-메서드)
5. [연습 문제](#5-연습-문제)

---

## 1. 문자열 슬라이스

**문자열 슬라이스**(`&str`)는 `String`(또는 문자열 리터럴)의 일부에 대한 참조입니다:

```rust
fn main() {
    let s = String::from("hello world");

    let hello = &s[0..5];   // "hello"
    let world = &s[6..11];  // "world"

    // Shorthand
    let hello = &s[..5];    // From start
    let world = &s[6..];    // To end
    let full = &s[..];      // Entire string

    println!("{hello} {world}");
}
```

```
String::from("hello world")

Stack:                          Heap:
s: [ptr, len=11, cap=11] ────→ [h|e|l|l|o| |w|o|r|l|d]
                                 0 1 2 3 4 5 6 7 8 9 10

hello: &s[0..5]
[ptr, len=5] ─────────────────→ [h|e|l|l|o]
                                 ↑ points into same heap memory

world: &s[6..11]
[ptr, len=5] ─────────────────→ [w|o|r|l|d]
                                 ↑ offset 6 into same allocation
```

### 1.1 UTF-8과 슬라이스 경계

Rust 문자열은 UTF-8로 인코딩됩니다. 멀티바이트 문자의 중간 바이트 경계에서 슬라이싱하면 패닉이 발생합니다:

```rust
fn main() {
    let emoji = String::from("🦀 Rust");
    // let slice = &emoji[0..2];  // PANIC: byte 2 is inside the 🦀 codepoint (4 bytes)
    let slice = &emoji[0..4];     // OK: "🦀" (complete codepoint)
    let rest = &emoji[5..];       // "Rust"

    // Safe alternatives for character-level operations:
    for ch in emoji.chars() {
        print!("{ch} ");  // 🦀   R u s t
    }
}
```

---

## 2. String vs &str

| 특성 | `String` | `&str` |
|------|----------|--------|
| 소유권(Ownership) | 소유 | 빌림 |
| 가변성(Mutability) | 가변 성장 가능(`push`, `push_str`) | 불변 뷰 |
| 저장소 | 힙 할당 | 힙, 스택, 또는 바이너리를 가리킴 |
| 크기 | ptr + len + capacity (24바이트) | ptr + len (16바이트) |
| 사용 사례 | 문자열 빌드/수정 | 문자열 읽기/전달 |

```rust
// String literals are &str — they live in the compiled binary
let literal: &str = "hello";

// String is heap-allocated and owned
let owned: String = String::from("hello");
let also_owned: String = "hello".to_string();

// &str from a String (cheap — just a pointer)
let slice: &str = &owned;
let slice: &str = owned.as_str();

// String from &str (allocates)
let new_string: String = literal.to_string();
let new_string: String = String::from(literal);
```

### 2.1 관용적인 함수 시그니처

```rust
// GOOD: accepts both String and &str
fn greet(name: &str) {
    println!("Hello, {name}!");
}

// Less flexible: only accepts String
fn greet_owned(name: String) {
    println!("Hello, {name}!");
}

fn main() {
    let owned = String::from("Alice");
    let literal = "Bob";

    greet(&owned);    // &String coerces to &str automatically (deref coercion)
    greet(literal);   // &str passed directly

    greet_owned(owned);    // Moves the String
    // greet_owned(literal);  // ERROR: expected String, found &str
}
```

> **원칙**: 함수 매개변수로는 `&str`을 받고, 호출자에게 소유권을 넘겨줄 필요가 있을 때는 `String`을 반환하세요.

---

## 3. 배열과 Vec 슬라이스

슬라이스는 배열과 `Vec<T>`에도 동작합니다:

```rust
fn sum(numbers: &[i32]) -> i32 {
    numbers.iter().sum()
}

fn main() {
    // Slice from array
    let arr = [1, 2, 3, 4, 5];
    let slice = &arr[1..4];  // [2, 3, 4]
    println!("sum of slice: {}", sum(slice));

    // Slice from Vec
    let vec = vec![10, 20, 30, 40, 50];
    let slice = &vec[..3];   // [10, 20, 30]
    println!("sum of vec slice: {}", sum(slice));

    // Entire collection as slice
    println!("sum of all: {}", sum(&arr));   // &[i32; 5] coerces to &[i32]
    println!("sum of all: {}", sum(&vec));   // &Vec<i32> coerces to &[i32]
}
```

### 3.1 가변 슬라이스

```rust
fn zero_out(data: &mut [i32]) {
    for element in data.iter_mut() {
        *element = 0;
    }
}

fn main() {
    let mut numbers = [1, 2, 3, 4, 5];
    zero_out(&mut numbers[1..4]);
    println!("{numbers:?}");  // [1, 0, 0, 0, 5]
}
```

---

## 4. 슬라이스 메서드

### 4.1 문자열 슬라이스 메서드

```rust
fn main() {
    let s = "Hello, Rust World!";

    // Searching
    println!("{}", s.contains("Rust"));      // true
    println!("{}", s.starts_with("Hello"));  // true
    println!("{:?}", s.find("Rust"));        // Some(7)

    // Splitting
    let words: Vec<&str> = s.split_whitespace().collect();
    println!("{words:?}");  // ["Hello,", "Rust", "World!"]

    let parts: Vec<&str> = "a,b,c".split(',').collect();
    println!("{parts:?}");  // ["a", "b", "c"]

    // Trimming
    let padded = "  hello  ";
    println!("'{}'", padded.trim());        // 'hello'
    println!("'{}'", padded.trim_start());  // 'hello  '

    // Replacing
    let replaced = s.replace("Rust", "Ferris");
    println!("{replaced}");  // Hello, Ferris World!

    // Case conversion (returns new String)
    println!("{}", s.to_uppercase());
    println!("{}", s.to_lowercase());
}
```

### 4.2 배열/Vec 슬라이스 메서드

```rust
fn main() {
    let data = [3, 1, 4, 1, 5, 9, 2, 6];

    // Searching
    println!("{}", data.contains(&5));           // true
    println!("{:?}", data.iter().position(|&x| x == 9)); // Some(5)

    // Windowing
    for window in data.windows(3) {
        print!("{window:?} ");  // [3,1,4] [1,4,1] [4,1,5] ...
    }
    println!();

    // Chunking
    for chunk in data.chunks(3) {
        print!("{chunk:?} ");  // [3,1,4] [1,5,9] [2,6]
    }
    println!();

    // Sorting (requires mutable slice)
    let mut sorted = data;
    sorted.sort();
    println!("{sorted:?}");  // [1, 1, 2, 3, 4, 5, 6, 9]

    // Binary search (on sorted data)
    println!("{:?}", sorted.binary_search(&4));  // Ok(4)
}
```

---

## 5. 연습 문제

### 연습 1: 첫 번째 단어
문자열에서 첫 번째 단어(첫 번째 공백 이전의 텍스트, 공백이 없으면 전체 문자열)를 반환하는 `fn first_word(s: &str) -> &str`를 작성하세요.

### 연습 2: 문자열 역순
단어의 순서를 뒤집는 `fn reverse_words(s: &str) -> String`을 작성하세요. `"hello world"` → `"world hello"`.

### 연습 3: 슬라이스 합계
`data.windows(window)`를 사용하여 이동 평균을 계산하는 `fn moving_average(data: &[f64], window: usize) -> Vec<f64>`를 작성하세요.

### 연습 4: &str vs String
`fn process(s: String)`에서 `fn process(s: &str)`로 함수를 리팩터링하세요. 리팩터링된 버전이 더 유연한 이유를 설명하세요.

### 연습 5: 안전한 부분 문자열
범위가 유효하지 않거나 UTF-8 경계에 걸리는 경우 패닉 대신 `None`을 반환하는 `fn safe_substring(s: &str, start: usize, end: usize) -> Option<&str>`를 작성하세요.

---

## 참고 자료
- [The Rust Book — The Slice Type](https://doc.rust-lang.org/book/ch04-03-slices.html)
- [std::str documentation](https://doc.rust-lang.org/std/primitive.str.html)
- [std::slice documentation](https://doc.rust-lang.org/std/primitive.slice.html)

---

**이전**: [빌림과 참조(Borrowing and References)](./04_Borrowing_and_References.md) | **다음**: [구조체와 메서드(Structs and Methods)](./06_Structs_and_Methods.md)
