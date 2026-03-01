# 17. 안전하지 않은 러스트(Unsafe Rust)

**이전**: [모듈과 카고](./16_Modules_and_Cargo.md) | **다음**: [프로젝트: CLI 도구](./18_Project_CLI_Tool.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Rust에서 `unsafe`의 의미를 설명하고 해제되는 다섯 가지 기능 식별하기
2. 원시 포인터(Raw Pointer) (`*const T`, `*mut T`)를 생성, 캐스팅(Casting), 역참조(Dereference)하기
3. 올바른 불변성(Invariant)으로 안전하지 않은 코드를 캡슐화하는 안전한 추상화(Safe Abstraction) 작성하기
4. FFI를 사용하여 Rust에서 C 함수를 호출하고 C에 Rust 함수를 노출하기
5. `unsafe`가 정당화되는 경우와 안전한 대안이 있는 경우 평가하기

---

"unsafe"라는 단어는 불길하게 들리지만, "이 코드는 잘못되었다"는 의미가 아닙니다. "컴파일러가 이 불변성을 검증할 수 없다 — 프로그래머가 책임을 진다"는 의미입니다. 안전한 Rust는 컴파일 타임에 강력한 보장을 제공합니다: 데이터 레이스(Data Race) 없음, 댕글링 포인터(Dangling Pointer) 없음, 버퍼 오버플로우(Buffer Overflow) 없음. 하지만 일부 정당한 작업들 — 하드웨어와 통신하기, C 라이브러리 호출하기, 고도로 최적화된 데이터 구조 구현하기 — 은 그러한 보장의 범위를 벗어나야 합니다. 그것이 바로 `unsafe`가 존재하는 이유입니다.

이렇게 생각하세요: 안전한 Rust는 가드레일이 있는 잘 표시된 도로에서 운전하는 것과 같습니다. 안전하지 않은 Rust는 오프로드(Off-Road)로 가는 것과 같습니다 — 차는 여전히 작동하지만, 어디로 운전하는지는 당신이 책임져야 합니다.

## 목차
1. [unsafe의 의미](#1-unsafe의-의미)
2. [다섯 가지 unsafe 초능력](#2-다섯-가지-unsafe-초능력)
3. [원시 포인터](#3-원시-포인터)
4. [안전하지 않은 함수 호출](#4-안전하지-않은-함수-호출)
5. [안전하지 않은 코드 위의 안전한 추상화](#5-안전하지-않은-코드-위의-안전한-추상화)
6. [FFI: Rust에서 C 호출하기](#6-ffi-rust에서-c-호출하기)
7. [FFI: C에서 Rust 호출하기](#7-ffi-c에서-rust-호출하기)
8. [안전하지 않은 트레이트](#8-안전하지-않은-트레이트)
9. [unsafe가 정당화되는 경우 vs 피할 수 있는 경우](#9-unsafe가-정당화되는-경우-vs-피할-수-있는-경우)
10. [연습 문제](#10-연습-문제)

---

## 1. unsafe의 의미

`unsafe`는 당신과 컴파일러 사이의 계약입니다:

```
┌─────────────────────────────────────────────────────┐
│                  SAFE RUST                           │
│  The compiler guarantees:                           │
│  ✓ No dangling references                          │
│  ✓ No data races                                   │
│  ✓ No null pointer dereferences                    │
│  ✓ No buffer overflows                             │
│  ✓ No use-after-free                               │
└────────────────────┬────────────────────────────────┘
                     │
              unsafe { ... }
                     │
┌────────────────────▼────────────────────────────────┐
│                UNSAFE RUST                           │
│  The compiler STILL enforces:                       │
│  ✓ Type checking                                   │
│  ✓ Borrow checking (on references)                 │
│  ✓ Lifetime checking (on references)               │
│                                                     │
│  But the programmer must guarantee:                 │
│  ✗ Raw pointers are valid when dereferenced         │
│  ✗ FFI contracts are honored                       │
│  ✗ Unsafe trait invariants are upheld              │
│  ✗ No undefined behavior                           │
└─────────────────────────────────────────────────────┘
```

중요한 오해: `unsafe`는 빌로우 체커(Borrow Checker)를 **비활성화하지 않습니다**. 일반 참조(`&T`, `&mut T`)는 `unsafe` 블록 안에서도 완전히 검사됩니다. 아래에 나열된 다섯 가지 특정 초능력만이 해제됩니다.

---

## 2. 다섯 가지 unsafe 초능력

```rust
fn main() {
    // 이 다섯 가지 연산은 `unsafe` 블록을 필요로 합니다:

    // 1. 원시 포인터 역참조
    let x = 42;
    let raw = &x as *const i32;
    unsafe { println!("raw: {}", *raw); }

    // 2. 안전하지 않은 함수나 메서드 호출
    unsafe { dangerous(); }

    // 3. 가변 정적 변수(Mutable Static Variable) 접근 또는 수정
    unsafe { COUNTER += 1; }

    // 4. 안전하지 않은 트레이트 구현
    //    (섹션 8 참고)

    // 5. 유니온(Union)의 필드 접근
    //    (아래 참고)
}

unsafe fn dangerous() {
    println!("You called an unsafe function!");
}

static mut COUNTER: u32 = 0;

// 유니온(Union): C 유니온처럼 겹치는 메모리 레이아웃.
// 잘못된 필드를 읽는 것은 미정의 동작(Undefined Behavior)입니다.
union IntOrFloat {
    i: i32,
    f: f32,
}

fn union_demo() {
    let value = IntOrFloat { i: 42 };
    // 컴파일러가 어떤 필드가 마지막으로 쓰여졌는지 알 수 없기 때문에
    // 유니온 필드를 읽으려면 unsafe가 필요합니다
    unsafe {
        println!("as int: {}", value.i);
        // 여기서 value.f를 읽으면 비트를 재해석합니다 — 합법적이지만
        // 결과는 의미 없을 수 있습니다
    }
}
```

---

## 3. 원시 포인터

원시 포인터(Raw Pointer) (`*const T`와 `*mut T`)는 Rust의 C 포인터에 해당합니다. 참조와 달리 원시 포인터는:

- null이 될 수 있습니다
- 댕글링(Dangling, 해제된 메모리를 가리킬 수 있음)이 될 수 있습니다
- 앨리어싱(Aliasing, 같은 데이터에 대한 여러 `*mut T`)이 될 수 있습니다
- 빌로우 체커가 추적하지 않습니다

### 3.1 원시 포인터 생성하기

원시 포인터를 생성하는 것은 항상 안전합니다. **역참조**만 `unsafe`가 필요합니다:

```rust
fn main() {
    let mut value = 10;

    // 참조에서 원시 포인터 생성 — 항상 안전
    let const_ptr: *const i32 = &value as *const i32;
    let mut_ptr: *mut i32 = &mut value as *mut i32;

    // 임의의 주소에 대한 포인터도 생성할 수 있습니다 (역참조하지 마세요!)
    let suspicious: *const i32 = 0x012345 as *const i32;

    // 포인터를 출력하면 메모리 주소가 표시됩니다 — 안전, 역참조 없음
    println!("const_ptr: {:p}", const_ptr);
    println!("mut_ptr:   {:p}", mut_ptr);
    println!("suspicious:{:p}", suspicious);

    // 역참조는 unsafe가 필요합니다
    unsafe {
        println!("*const_ptr = {}", *const_ptr);  // 10
        *mut_ptr = 20;                              // 원시 포인터를 통해 수정
        println!("*const_ptr = {}", *const_ptr);   // 20
    }
}
```

### 3.2 포인터 산술(Pointer Arithmetic)

```rust
fn main() {
    let data = [10, 20, 30, 40, 50];
    let ptr = data.as_ptr(); // 첫 번째 원소를 가리키는 *const i32

    unsafe {
        // .add(n)은 n개의 원소만큼 전진합니다 (바이트가 아님)
        for i in 0..data.len() {
            // 포인터를 오프셋한 다음 역참조
            let val = *ptr.add(i);
            println!("data[{i}] = {val}");
        }
    }
}
```

### 3.3 널 포인터(Null Pointer)

```rust
use std::ptr;

fn main() {
    // Rust에는 명시적인 널 포인터 생성자가 있습니다
    let null_ptr: *const i32 = ptr::null();
    let null_mut: *mut i32 = ptr::null_mut();

    // 역참조 전에 널인지 확인
    if null_ptr.is_null() {
        println!("Pointer is null — will not dereference");
    }

    // 널 역참조는 미정의 동작입니다 — 절대 하지 마세요
    // unsafe { println!("{}", *null_ptr); }  // UB!
}
```

---

## 4. 안전하지 않은 함수 호출

일부 표준 라이브러리 함수는 컴파일러가 검사할 수 없는 전제조건(Precondition)이 있기 때문에 `unsafe`로 표시됩니다:

```rust
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];

    // 안전한 버전: 인덱스가 범위를 벗어나면 패닉(Panic)
    let third = v[2];

    // 안전하지 않은 버전: 범위 검사 없음 — 범위를 벗어나면 미정의 동작
    // 이미 인덱스를 검증한 타이트한 루프에서 유용합니다
    unsafe {
        let third = *v.get_unchecked(2);
        println!("third: {third}");
    }

    // 또 다른 일반적인 예: 원시 바이트로 String 생성
    let bytes = vec![72, 101, 108, 108, 111]; // ASCII로 "Hello"
    unsafe {
        // from_utf8_unchecked는 UTF-8 검증을 건너뜁니다.
        // 호출자는 바이트가 유효한 UTF-8임을 보장해야 합니다.
        let greeting = String::from_utf8_unchecked(bytes);
        println!("{greeting}");
    }

    // 안전한 대안: from_utf8은 검증하고 Result를 반환합니다
    let bytes2 = vec![72, 101, 108, 108, 111];
    let greeting2 = String::from_utf8(bytes2).expect("valid UTF-8");
    println!("{greeting2}");
}
```

### 자신만의 unsafe 함수 작성하기

```rust
/// 범위 검사 없이 `index`의 원소를 반환합니다.
///
/// # Safety
///
/// 호출자는 `index < slice.len()`임을 보장해야 합니다.
unsafe fn get_unchecked<T>(slice: &[T], index: usize) -> &T {
    // # Safety 섹션에 전제조건을 문서화합니다.
    // 이 함수를 호출하는 모든 코드는 이를 지켜야 합니다.
    &*slice.as_ptr().add(index)
}

fn main() {
    let data = [10, 20, 30];

    // 호출자가 인덱스의 유효성에 대한 책임을 집니다
    unsafe {
        let val = get_unchecked(&data, 1);
        println!("data[1] = {val}");

        // 이것은 UB입니다 — 인덱스 범위 초과:
        // let bad = get_unchecked(&data, 100);
    }
}
```

---

## 5. 안전하지 않은 코드 위의 안전한 추상화

unsafe Rust에서 가장 중요한 패턴: 내부적으로 `unsafe`를 사용하지만 안전한 공개 API를 노출합니다. 표준 라이브러리는 이것을 광범위하게 합니다 — `Vec`, `String`, `HashMap` 모두 내부적으로 `unsafe`를 사용합니다.

```rust
/// 고정 크기 버퍼(Buffer)의 간단한 래퍼(Wrapper).
/// unsafe 코드는 캡슐화됩니다 — 사용자는 안전한 메서드를 통해서만 상호작용합니다.
struct FixedBuffer {
    data: [u8; 1024],
    len: usize,
}

impl FixedBuffer {
    fn new() -> Self {
        FixedBuffer {
            data: [0u8; 1024],
            len: 0,
        }
    }

    /// 버퍼에 바이트를 추가합니다. 가득 차면 false를 반환합니다.
    fn push(&mut self, byte: u8) -> bool {
        if self.len >= self.data.len() {
            return false;
        }
        // 안전한 추상화: 위에서 len < capacity를 검증했으므로
        // 검사 없는 쓰기는 유효합니다.
        unsafe {
            *self.data.get_unchecked_mut(self.len) = byte;
        }
        self.len += 1;
        true
    }

    /// 인덱스의 바이트를 가져옵니다. 범위를 벗어나면 None을 반환합니다.
    fn get(&self, index: usize) -> Option<u8> {
        if index >= self.len {
            return None;
        }
        // 안전한 추상화: 위에서 인덱스를 검증했습니다
        unsafe { Some(*self.data.get_unchecked(index)) }
    }

    fn len(&self) -> usize {
        self.len
    }
}

fn main() {
    let mut buf = FixedBuffer::new();
    buf.push(b'H');
    buf.push(b'i');

    // 사용자는 `unsafe`를 보거나 작성할 필요가 없습니다 — API는 완전히 안전합니다
    println!("buf[0] = {:?}", buf.get(0));   // Some(72)
    println!("buf[5] = {:?}", buf.get(5));   // None — 안전, 패닉 없음
    println!("length = {}", buf.len());       // 2
}
```

### `split_at_mut` 패턴

표준 라이브러리의 고전적인 예 — 가변 슬라이스(Slice)를 겹치지 않는 두 개의 가변 슬라이스로 분리하려면, 빌로우 체커가 같은 데이터에 대한 두 개의 `&mut` 빌로우를 보기 때문에 unsafe가 필요합니다:

```rust
fn split_at_mut_demo(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = slice.len();
    assert!(mid <= len, "mid out of bounds");

    let ptr = slice.as_mut_ptr();

    unsafe {
        // 이것이 안전한 이유:
        // 1. mid <= len을 검증했으므로 두 서브슬라이스 모두 범위 내에 있습니다
        // 2. 두 슬라이스는 겹치지 않습니다 (하나는 [0..mid), 다른 하나는 [mid..len))
        // 3. 빌로우 체커가 독립적으로 추적할 두 개의 &mut 슬라이스를 반환합니다
        (
            std::slice::from_raw_parts_mut(ptr, mid),
            std::slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

fn main() {
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let (left, right) = split_at_mut_demo(&mut data, 3);
    left[0] = 100;
    right[0] = 400;
    println!("left: {left:?}");   // [100, 2, 3]
    println!("right: {right:?}"); // [400, 5, 6]
}
```

---

## 6. FFI: Rust에서 C 호출하기

FFI(Foreign Function Interface, 외부 함수 인터페이스)를 사용하면 Rust가 C(또는 C 호출 규약을 사용하는 모든 언어)로 작성된 함수를 호출할 수 있습니다. 이것이 프로덕션(Production) 코드에서 `unsafe`의 주요 이유입니다 — 운영 체제, 하드웨어, 광대한 C 라이브러리 생태계와 인터페이스하기 위해서입니다.

### 6.1 libc 함수 호출하기

```rust
// `extern "C"` 블록에서 외부 C 함수를 선언합니다.
// "C"는 호출 규약(Calling Convention, 인수를 전달하는 방법,
// 스택을 관리하는 방법 등)을 지정합니다
extern "C" {
    fn abs(input: i32) -> i32;
    fn sqrt(input: f64) -> f64;
}

fn main() {
    // 모든 FFI 호출은 unsafe입니다. 컴파일러가 다음을 검증할 수 없기 때문입니다:
    // - 함수 시그니처(Signature)가 실제 C 함수와 일치하는지
    // - 함수가 메모리 안전성을 위반하지 않는지
    // - 함수가 모든 입력을 올바르게 처리하는지
    unsafe {
        println!("abs(-5) = {}", abs(-5));         // 5
        println!("sqrt(2.0) = {}", sqrt(2.0));     // 1.4142...
    }
}
```

### 6.2 C 라이브러리에 링크하기

다음 헤더를 가진 C 라이브러리가 있다고 가정합니다:

```c
// mathlib.h
int add(int a, int b);
double circle_area(double radius);
```

```rust
// Rust 쪽: 외부 함수를 선언하고 Cargo에 링크 방법을 알립니다

// #[link(name = "mathlib")]은 링커가 libmathlib.a 또는 libmathlib.so를 찾도록 합니다
#[link(name = "m")]  // libm(표준 수학 라이브러리)에 링크
extern "C" {
    fn pow(base: f64, exp: f64) -> f64;
    fn log2(x: f64) -> f64;
}

fn main() {
    unsafe {
        println!("2^10 = {}", pow(2.0, 10.0));  // 1024.0
        println!("log2(256) = {}", log2(256.0)); // 8.0
    }
}
```

### 6.3 C 문자열 다루기

C 문자열은 널(Null) 종료 `*const c_char`이고, Rust 문자열은 길이 접두사가 붙은 UTF-8입니다. `std::ffi` 모듈이 이 격차를 해소합니다:

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

extern "C" {
    fn strlen(s: *const c_char) -> usize;
}

fn main() {
    // Rust String → C 문자열
    // CString::new는 널 종료자를 추가하고 내부 널을 검사합니다
    let rust_string = "Hello from Rust";
    let c_string = CString::new(rust_string).expect("CString creation failed");

    unsafe {
        // .as_ptr()는 C 함수에 적합한 *const c_char를 제공합니다
        let len = strlen(c_string.as_ptr());
        println!("C says length is: {len}");  // 15
    }

    // C 문자열 → Rust &str
    unsafe {
        let c_ptr = c_string.as_ptr();
        // CStr::from_ptr는 기존의 널 종료 C 문자열을 래핑합니다
        let back_to_rust: &str = CStr::from_ptr(c_ptr)
            .to_str()
            .expect("Invalid UTF-8");
        println!("Back in Rust: {back_to_rust}");
    }
}
```

### 6.4 repr(C) — C 메모리 레이아웃 맞추기

```rust
// #[repr(C)]는 Rust가 C 컴파일러와 동일한 순서와 정렬로 필드를 배치하도록 보장합니다.
// 이 없으면 Rust는 효율성을 위해 필드를 재정렬할 수 있습니다.
#[repr(C)]
struct Point {
    x: f64,
    y: f64,
}

extern "C" {
    // Point를 포인터로 받는 가상의 C 함수
    // fn distance_from_origin(p: *const Point) -> f64;
}

fn main() {
    let p = Point { x: 3.0, y: 4.0 };
    // #[repr(C)]로, &p를 동일한 레이아웃의 구조체를 기대하는 C 코드에 안전하게 전달 가능
    println!("Point at ({}, {})", p.x, p.y);
}
```

---

## 7. FFI: C에서 Rust 호출하기

C 코드가 Rust 함수를 호출할 수 있도록 노출할 수도 있습니다:

```rust
// #[no_mangle]은 Rust가 함수 이름을 맹글링(Mangling)하지 않도록 합니다.
// 이것이 없으면 컴파일된 심볼이 다음과 같을 수 있습니다:
// _ZN7mylib3add17h1234567890abcdefE — C는 절대 찾을 수 없습니다.
#[no_mangle]
pub extern "C" fn rust_add(a: i32, b: i32) -> i32 {
    a + b
}

// 문자열의 경우 C 호환 포인터를 반환합니다
#[no_mangle]
pub extern "C" fn rust_greeting() -> *const u8 {
    // 문자열 리터럴이 'static 라이프타임(Lifetime)을 가지기 때문에 안전합니다.
    // 프로그램 전체 기간 동안 존재합니다 — 댕글링 포인터 없음.
    b"Hello from Rust!\0".as_ptr()
}
```

C 쪽은 다음과 같습니다:

```c
// main.c
#include <stdio.h>

// Rust 함수 선언
extern int rust_add(int a, int b);
extern const char* rust_greeting();

int main() {
    printf("3 + 4 = %d\n", rust_add(3, 4));
    printf("%s\n", rust_greeting());
    return 0;
}
```

빌드하려면 Rust 코드를 정적 또는 동적 라이브러리로 컴파일합니다:

```toml
# Cargo.toml
[lib]
name = "mylib"
crate-type = ["cdylib"]  # libmylib.so / libmylib.dylib / mylib.dll을 생성합니다
```

```bash
cargo build --release
# 그런 다음 C에서 링크:
gcc main.c -L target/release -lmylib -o main
```

---

## 8. 안전하지 않은 트레이트

일부 트레이트(Trait)는 컴파일러가 검증할 수 없는 불변성을 가집니다. 이를 구현하려면 `unsafe impl`이 필요합니다:

```rust
// Send: 타입을 다른 스레드로 전송할 수 있습니다
// Sync: 타입을 스레드 간에 (&T를 통해) 공유할 수 있습니다
//
// 대부분의 타입은 Send와 Sync를 자동으로 구현합니다.
// 원시 포인터나 다른 unsafe 프리미티브를 사용하는 타입에 대해서만
// `unsafe impl`이 필요합니다.

struct MyBox {
    ptr: *mut i32,
}

// 기본적으로 *mut i32는 Send도 Sync도 아닙니다.
// 가리키는 데이터의 독점적 소유권을 보장하기 때문에
// 우리의 타입이 스레드 간에 안전하게 전송될 수 있다고 단언합니다.
unsafe impl Send for MyBox {}
unsafe impl Sync for MyBox {}

impl MyBox {
    fn new(value: i32) -> Self {
        let layout = std::alloc::Layout::new::<i32>();
        unsafe {
            let ptr = std::alloc::alloc(layout) as *mut i32;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            ptr.write(value);
            MyBox { ptr }
        }
    }

    fn get(&self) -> i32 {
        unsafe { *self.ptr }
    }
}

impl Drop for MyBox {
    fn drop(&mut self) {
        unsafe {
            let layout = std::alloc::Layout::new::<i32>();
            std::alloc::dealloc(self.ptr as *mut u8, layout);
        }
    }
}

fn main() {
    let b = MyBox::new(42);
    println!("MyBox holds: {}", b.get());

    // Send를 구현했기 때문에 다른 스레드로 이동할 수 있습니다
    let handle = std::thread::spawn(move || {
        println!("In thread: {}", b.get());
    });
    handle.join().unwrap();
}
```

---

## 9. unsafe가 정당화되는 경우 vs 피할 수 있는 경우

### 정당화되는 사용

| 시나리오 | unsafe가 필요한 이유 |
|---------|-------------------|
| FFI (C/C++ 라이브러리 호출) | 컴파일러가 외부 코드를 검증할 수 없음 |
| 커스텀 할당자(Custom Allocator) | 수동 메모리 관리 |
| 락-프리(Lock-Free) 데이터 구조 | 원자적 연산 + 원시 포인터 |
| SIMD 인트린식(Intrinsic) | CPU 특정 명령어 |
| `split_at_mut` 스타일 패턴 | 빌로우 체커가 너무 보수적 |
| OS/하드웨어 상호작용 | 원시 메모리 매핑 I/O |

### 피할 수 있는 사용

| 유혹 | 안전한 대안 |
|-----|-----------|
| 성능을 위한 `*ptr` | 이터레이터(Iterator)는 보통 마찬가지로 빠릅니다 — 먼저 확인하세요 |
| 어디서나 `get_unchecked` | 일반 인덱싱 사용; 최적화 전에 프로파일링하세요 |
| 타입 변환을 위한 `transmute` | `From`/`Into` 트레이트, `as` 캐스트 |
| 전역 상태를 위한 `static mut` | `std::sync::OnceLock`, `Mutex`, `AtomicU32` |
| 내부 가변성(Interior Mutability) | `Cell`, `RefCell`, `Mutex`, `RwLock` |

### 모범 사례

```
1. unsafe 범위 최소화
   나쁨:  unsafe { ... 50줄 ... }
   좋음: unsafe { single_operation() }

2. # Safety로 불변성 문서화
   /// # Safety
   /// `ptr`은 널이 아니어야 하고 유효한 `Foo`를 가리켜야 합니다.

3. 안전한 API로 unsafe를 래핑
   사용자가 당신의 라이브러리를 사용하기 위해 `unsafe`를 작성할 필요가 없어야 합니다.

4. 도구를 사용하여 unsafe 코드 검증
   - cargo miri test      (런타임에 미정의 동작 감지)
   - cargo careful test   (추가 런타임 검사)
   - unsafe 패턴에 대한 clippy 린트
```

---

## 10. 연습 문제

### 연습 1: 안전한 래퍼
내부적으로 `*mut T`와 `len: usize`를 저장하는 `SafeArray<T>` 구조체를 작성하세요. `new(size: usize, default: T)` (할당), `get(index: usize) -> Option<&T>`, `set(index: usize, value: T) -> bool`, `Drop`을 구현하세요. 모든 공개 메서드는 안전해야 합니다. 메모리 관리에 `std::alloc::alloc`과 `std::alloc::dealloc`을 사용하세요.

### 연습 2: C 문자열 변환기
Rust `&str`을 `CString`으로 변환하고, FFI를 통해 C의 `strlen`을 호출하고, 결과를 반환하는 함수 `fn safe_strlen(s: &str) -> usize`를 작성하세요. 입력 문자열에 내부 널 바이트가 포함된 경우를 처리하세요 (그 경우 0을 반환).

### 연습 3: 재해석 캐스트(Reinterpret Cast)
`f32::to_bits()`를 사용하지 않고 float의 원시 IEEE 754 비트 패턴을 반환하는 함수 `fn f32_to_bits(f: f32) -> u32`를 작성하세요. 원시 포인터 캐스트를 사용하세요. 그런 다음 역방향을 위한 `fn bits_to_f32(bits: u32) -> f32`를 작성하세요. 검증: `f32_to_bits(1.0)`은 `0x3F800000`과 같아야 합니다.

### 연습 4: 안전한 split_at_mut
`&mut [i32]`에 대한 자신만의 `split_at_mut` 버전을 구현하세요. `mid > len`이면 패닉이 발생해야 합니다. 다음을 검증하는 테스트를 작성하세요: (a) 두 절반을 독립적으로 수정할 수 있음, (b) 범위를 벗어난 mid에 패닉, (c) 빈 슬라이스와 mid==0, mid==len을 처리.

---

## 참고 자료
- [The Rust Book — Unsafe Rust](https://doc.rust-lang.org/book/ch20-01-unsafe-rust.html)
- [The Rustonomicon](https://doc.rust-lang.org/nomicon/) — unsafe Rust의 결정판 가이드
- [Rust FFI 가이드](https://doc.rust-lang.org/nomicon/ffi.html)
- [std::ffi 모듈](https://doc.rust-lang.org/std/ffi/index.html)
- [Miri — 미정의 동작 감지기](https://github.com/rust-lang/miri)

---

**이전**: [모듈과 카고](./16_Modules_and_Cargo.md) | **다음**: [프로젝트: CLI 도구](./18_Project_CLI_Tool.md)
