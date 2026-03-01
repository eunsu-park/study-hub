# 17. Unsafe Rust

**Previous**: [Modules and Cargo](./16_Modules_and_Cargo.md) | **Next**: [Project: CLI Tool](./18_Project_CLI_Tool.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what `unsafe` means in Rust and identify the five capabilities it unlocks
2. Create, cast, and dereference raw pointers (`*const T`, `*mut T`)
3. Write safe abstractions that encapsulate unsafe code with correct invariants
4. Call C functions from Rust and expose Rust functions to C using FFI
5. Evaluate when `unsafe` is justified versus when a safe alternative exists

---

The word "unsafe" sounds alarming, but it does not mean "this code is broken." It means "the compiler cannot verify these invariants — the programmer takes responsibility." Safe Rust provides strong guarantees at compile time: no data races, no dangling pointers, no buffer overflows. But some legitimate operations — talking to hardware, calling C libraries, implementing highly optimized data structures — require stepping outside those guarantees. That is what `unsafe` is for.

Think of it this way: safe Rust is like driving on well-marked roads with guardrails. Unsafe Rust is like going off-road — the car still works, but you are responsible for watching where you drive.

## Table of Contents
1. [What Unsafe Means](#1-what-unsafe-means)
2. [The Five Unsafe Superpowers](#2-the-five-unsafe-superpowers)
3. [Raw Pointers](#3-raw-pointers)
4. [Calling Unsafe Functions](#4-calling-unsafe-functions)
5. [Safe Abstractions over Unsafe Code](#5-safe-abstractions-over-unsafe-code)
6. [FFI: Calling C from Rust](#6-ffi-calling-c-from-rust)
7. [FFI: Calling Rust from C](#7-ffi-calling-rust-from-c)
8. [Unsafe Traits](#8-unsafe-traits)
9. [When Unsafe Is Justified vs Avoidable](#9-when-unsafe-is-justified-vs-avoidable)
10. [Practice Problems](#10-practice-problems)

---

## 1. What Unsafe Means

`unsafe` is a contract between you and the compiler:

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

A critical misconception: `unsafe` does **not** disable the borrow checker. Regular references (`&T`, `&mut T`) are still fully checked inside `unsafe` blocks. Only the five specific superpowers listed below are unlocked.

---

## 2. The Five Unsafe Superpowers

```rust
fn main() {
    // These five operations require an `unsafe` block:

    // 1. Dereference a raw pointer
    let x = 42;
    let raw = &x as *const i32;
    unsafe { println!("raw: {}", *raw); }

    // 2. Call an unsafe function or method
    unsafe { dangerous(); }

    // 3. Access or modify a mutable static variable
    unsafe { COUNTER += 1; }

    // 4. Implement an unsafe trait
    //    (shown in section 8)

    // 5. Access fields of a union
    //    (shown below)
}

unsafe fn dangerous() {
    println!("You called an unsafe function!");
}

static mut COUNTER: u32 = 0;

// Unions: overlapping memory layout, like C unions.
// Reading the wrong field is undefined behavior.
union IntOrFloat {
    i: i32,
    f: f32,
}

fn union_demo() {
    let value = IntOrFloat { i: 42 };
    // Reading a union field requires unsafe because the compiler
    // cannot know which field was last written
    unsafe {
        println!("as int: {}", value.i);
        // Reading value.f here would reinterpret the bits — legal but
        // the result may be meaningless
    }
}
```

---

## 3. Raw Pointers

Raw pointers (`*const T` and `*mut T`) are Rust's equivalent of C pointers. Unlike references, they:

- Can be null
- Can be dangling (point to freed memory)
- Can alias (multiple `*mut T` to the same data)
- Are not tracked by the borrow checker

### 3.1 Creating Raw Pointers

Creating a raw pointer is always safe. Only **dereferencing** requires `unsafe`:

```rust
fn main() {
    let mut value = 10;

    // Create raw pointers from references — always safe
    let const_ptr: *const i32 = &value as *const i32;
    let mut_ptr: *mut i32 = &mut value as *mut i32;

    // You can even create a pointer to an arbitrary address (don't deref this!)
    let suspicious: *const i32 = 0x012345 as *const i32;

    // Printing pointers shows memory addresses — safe, no dereference
    println!("const_ptr: {:p}", const_ptr);
    println!("mut_ptr:   {:p}", mut_ptr);
    println!("suspicious:{:p}", suspicious);

    // Dereferencing requires unsafe
    unsafe {
        println!("*const_ptr = {}", *const_ptr);  // 10
        *mut_ptr = 20;                              // modify through raw pointer
        println!("*const_ptr = {}", *const_ptr);   // 20
    }
}
```

### 3.2 Pointer Arithmetic

```rust
fn main() {
    let data = [10, 20, 30, 40, 50];
    let ptr = data.as_ptr(); // *const i32 pointing to first element

    unsafe {
        // .add(n) advances by n elements (not bytes)
        for i in 0..data.len() {
            // offset the pointer, then dereference
            let val = *ptr.add(i);
            println!("data[{i}] = {val}");
        }
    }
}
```

### 3.3 Null Pointers

```rust
use std::ptr;

fn main() {
    // Rust has explicit null pointer constructors
    let null_ptr: *const i32 = ptr::null();
    let null_mut: *mut i32 = ptr::null_mut();

    // Check for null before dereferencing
    if null_ptr.is_null() {
        println!("Pointer is null — will not dereference");
    }

    // Dereferencing null is undefined behavior — never do this
    // unsafe { println!("{}", *null_ptr); }  // UB!
}
```

---

## 4. Calling Unsafe Functions

Some standard library functions are marked `unsafe` because they have preconditions the compiler cannot check:

```rust
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];

    // Safe version: panics if index is out of bounds
    let third = v[2];

    // Unsafe version: no bounds check — undefined behavior if out of bounds
    // Useful in tight loops where you have already validated the index
    unsafe {
        let third = *v.get_unchecked(2);
        println!("third: {third}");
    }

    // Another common example: constructing a String from raw bytes
    let bytes = vec![72, 101, 108, 108, 111]; // "Hello" in ASCII
    unsafe {
        // from_utf8_unchecked skips the UTF-8 validation.
        // The caller must guarantee the bytes are valid UTF-8.
        let greeting = String::from_utf8_unchecked(bytes);
        println!("{greeting}");
    }

    // Safe alternative: from_utf8 validates and returns Result
    let bytes2 = vec![72, 101, 108, 108, 111];
    let greeting2 = String::from_utf8(bytes2).expect("valid UTF-8");
    println!("{greeting2}");
}
```

### Writing Your Own Unsafe Functions

```rust
/// Returns the element at `index` without bounds checking.
///
/// # Safety
///
/// The caller must ensure that `index < slice.len()`.
unsafe fn get_unchecked<T>(slice: &[T], index: usize) -> &T {
    // We document the precondition in a # Safety section.
    // Anyone calling this function must uphold it.
    &*slice.as_ptr().add(index)
}

fn main() {
    let data = [10, 20, 30];

    // The caller takes responsibility for the validity of the index
    unsafe {
        let val = get_unchecked(&data, 1);
        println!("data[1] = {val}");

        // This would be UB — index out of bounds:
        // let bad = get_unchecked(&data, 100);
    }
}
```

---

## 5. Safe Abstractions over Unsafe Code

The most important pattern in unsafe Rust: use `unsafe` internally but expose a safe public API. The standard library does this extensively — `Vec`, `String`, `HashMap` all use `unsafe` internally.

```rust
/// A simple wrapper around a fixed-size buffer.
/// Unsafe code is encapsulated — users interact through safe methods only.
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

    /// Push a byte into the buffer. Returns false if full.
    fn push(&mut self, byte: u8) -> bool {
        if self.len >= self.data.len() {
            return false;
        }
        // Safe abstraction: we verified len < capacity above,
        // so the unchecked write is valid.
        unsafe {
            *self.data.get_unchecked_mut(self.len) = byte;
        }
        self.len += 1;
        true
    }

    /// Get a byte at an index. Returns None if out of bounds.
    fn get(&self, index: usize) -> Option<u8> {
        if index >= self.len {
            return None;
        }
        // Safe abstraction: we validated the index above
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

    // Users never see or write `unsafe` — the API is fully safe
    println!("buf[0] = {:?}", buf.get(0));   // Some(72)
    println!("buf[5] = {:?}", buf.get(5));   // None — safe, no panic
    println!("length = {}", buf.len());       // 2
}
```

### The `split_at_mut` Pattern

A classic example from the standard library — splitting a mutable slice into two non-overlapping mutable slices requires unsafe because the borrow checker sees two `&mut` borrows of the same data:

```rust
fn split_at_mut_demo(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = slice.len();
    assert!(mid <= len, "mid out of bounds");

    let ptr = slice.as_mut_ptr();

    unsafe {
        // This is safe because:
        // 1. We verified mid <= len, so both sub-slices are within bounds
        // 2. The two slices do not overlap (one is [0..mid), other is [mid..len))
        // 3. We return two &mut slices that the borrow checker will track independently
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

## 6. FFI: Calling C from Rust

FFI (Foreign Function Interface) lets Rust call functions written in C (or any language that uses the C calling convention). This is the primary reason for `unsafe` in production code — interfacing with operating systems, hardware, and the vast ecosystem of C libraries.

### 6.1 Calling libc Functions

```rust
// Declare external C functions in an `extern "C"` block.
// The "C" specifies the calling convention (how arguments are passed,
// how the stack is managed, etc.)
extern "C" {
    fn abs(input: i32) -> i32;
    fn sqrt(input: f64) -> f64;
}

fn main() {
    // All FFI calls are unsafe because the compiler cannot verify:
    // - The function signature matches the actual C function
    // - The function does not violate memory safety
    // - The function handles all inputs correctly
    unsafe {
        println!("abs(-5) = {}", abs(-5));         // 5
        println!("sqrt(2.0) = {}", sqrt(2.0));     // 1.4142...
    }
}
```

### 6.2 Linking to a C Library

Suppose you have a C library with this header:

```c
// mathlib.h
int add(int a, int b);
double circle_area(double radius);
```

```rust
// Rust side: declare the external functions and tell Cargo how to link

// #[link(name = "mathlib")] tells the linker to look for libmathlib.a or libmathlib.so
#[link(name = "m")]  // link to libm (standard math library)
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

### 6.3 Working with C Strings

C strings are null-terminated `*const c_char`, while Rust strings are length-prefixed UTF-8. The `std::ffi` module bridges the gap:

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

extern "C" {
    fn strlen(s: *const c_char) -> usize;
}

fn main() {
    // Rust String → C string
    // CString::new adds a null terminator and checks for interior nulls
    let rust_string = "Hello from Rust";
    let c_string = CString::new(rust_string).expect("CString creation failed");

    unsafe {
        // .as_ptr() gives a *const c_char suitable for C functions
        let len = strlen(c_string.as_ptr());
        println!("C says length is: {len}");  // 15
    }

    // C string → Rust &str
    unsafe {
        let c_ptr = c_string.as_ptr();
        // CStr::from_ptr wraps an existing null-terminated C string
        let back_to_rust: &str = CStr::from_ptr(c_ptr)
            .to_str()
            .expect("Invalid UTF-8");
        println!("Back in Rust: {back_to_rust}");
    }
}
```

### 6.4 repr(C) — Matching C Memory Layout

```rust
// #[repr(C)] ensures Rust lays out fields in the same order and alignment
// as a C compiler would. Without it, Rust may reorder fields for efficiency.
#[repr(C)]
struct Point {
    x: f64,
    y: f64,
}

extern "C" {
    // Hypothetical C function that takes a Point by pointer
    // fn distance_from_origin(p: *const Point) -> f64;
}

fn main() {
    let p = Point { x: 3.0, y: 4.0 };
    // With #[repr(C)], &p can safely be passed to C code expecting
    // a struct with the same layout
    println!("Point at ({}, {})", p.x, p.y);
}
```

---

## 7. FFI: Calling Rust from C

You can also expose Rust functions so C code can call them:

```rust
// #[no_mangle] prevents Rust from mangling the function name.
// Without it, the compiled symbol might be something like
// _ZN7mylib3add17h1234567890abcdefE — C could never find it.
#[no_mangle]
pub extern "C" fn rust_add(a: i32, b: i32) -> i32 {
    a + b
}

// For strings, return a C-compatible pointer
#[no_mangle]
pub extern "C" fn rust_greeting() -> *const u8 {
    // This is safe only because the string literal has 'static lifetime.
    // It lives for the entire program — no dangling pointer.
    b"Hello from Rust!\0".as_ptr()
}
```

The C side would look like:

```c
// main.c
#include <stdio.h>

// Declare the Rust functions
extern int rust_add(int a, int b);
extern const char* rust_greeting();

int main() {
    printf("3 + 4 = %d\n", rust_add(3, 4));
    printf("%s\n", rust_greeting());
    return 0;
}
```

To build, you compile the Rust code as a static or dynamic library:

```toml
# Cargo.toml
[lib]
name = "mylib"
crate-type = ["cdylib"]  # produces libmylib.so / libmylib.dylib / mylib.dll
```

```bash
cargo build --release
# Then link from C:
gcc main.c -L target/release -lmylib -o main
```

---

## 8. Unsafe Traits

Some traits have invariants that the compiler cannot verify. Implementing them requires `unsafe impl`:

```rust
// Send: a type can be transferred to another thread
// Sync: a type can be shared (via &T) between threads
//
// Most types implement Send and Sync automatically.
// You only need `unsafe impl` for types that use raw pointers
// or other unsafe primitives.

struct MyBox {
    ptr: *mut i32,
}

// By default, *mut i32 is neither Send nor Sync.
// We assert that our type is safe to send across threads
// because we guarantee exclusive ownership of the pointed-to data.
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

    // Because we implemented Send, we can move it to another thread
    let handle = std::thread::spawn(move || {
        println!("In thread: {}", b.get());
    });
    handle.join().unwrap();
}
```

---

## 9. When Unsafe Is Justified vs Avoidable

### Justified Uses

| Scenario | Why unsafe is needed |
|----------|---------------------|
| FFI (calling C/C++ libraries) | Compiler cannot verify foreign code |
| Custom allocators | Manual memory management |
| Lock-free data structures | Atomic operations + raw pointers |
| SIMD intrinsics | CPU-specific instructions |
| `split_at_mut`-style patterns | Borrow checker is too conservative |
| OS/hardware interaction | Raw memory-mapped I/O |

### Avoidable Uses

| Temptation | Safe Alternative |
|------------|-----------------|
| `*ptr` for performance | Iterators are usually just as fast — check first |
| `get_unchecked` everywhere | Use normal indexing; profile before optimizing |
| `transmute` to convert types | `From`/`Into` traits, `as` casts |
| `static mut` for global state | `std::sync::OnceLock`, `Mutex`, `AtomicU32` |
| Interior mutability | `Cell`, `RefCell`, `Mutex`, `RwLock` |

### Best Practices

```
1. Minimize unsafe scope
   Bad:  unsafe { ... 50 lines ... }
   Good: unsafe { single_operation() }

2. Document invariants with # Safety
   /// # Safety
   /// `ptr` must be non-null and point to a valid `Foo`.

3. Wrap unsafe in a safe API
   Users should never need to write `unsafe` to use your library.

4. Use tools to verify unsafe code
   - cargo miri test      (detects undefined behavior at runtime)
   - cargo careful test   (extra runtime checks)
   - clippy lints for unsafe patterns
```

---

## 10. Practice Problems

### Exercise 1: Safe Wrapper
Write a `SafeArray<T>` struct that internally stores a `*mut T` and a `len: usize`. Implement `new(size: usize, default: T)` (allocates), `get(index: usize) -> Option<&T>`, `set(index: usize, value: T) -> bool`, and `Drop`. All public methods must be safe. Use `std::alloc::alloc` and `std::alloc::dealloc` for memory management.

### Exercise 2: C String Converter
Write a function `fn safe_strlen(s: &str) -> usize` that converts a Rust `&str` to a `CString`, calls C's `strlen` via FFI, and returns the result. Handle the case where the input string contains an interior null byte (return 0 in that case).

### Exercise 3: Reinterpret Cast
Write a function `fn f32_to_bits(f: f32) -> u32` that returns the raw IEEE 754 bit pattern of a float **without** using `f32::to_bits()`. Use a raw pointer cast. Then write `fn bits_to_f32(bits: u32) -> f32` for the reverse. Verify: `f32_to_bits(1.0)` should equal `0x3F800000`.

### Exercise 4: Safe split_at_mut
Implement your own version of `split_at_mut` for `&mut [i32]`. It should panic if `mid > len`. Write tests that verify: (a) both halves can be mutated independently, (b) panics on out-of-bounds mid, (c) handles empty slices and mid==0 and mid==len.

---

## References
- [The Rust Book — Unsafe Rust](https://doc.rust-lang.org/book/ch20-01-unsafe-rust.html)
- [The Rustonomicon](https://doc.rust-lang.org/nomicon/) — the definitive guide to unsafe Rust
- [Rust FFI Guide](https://doc.rust-lang.org/nomicon/ffi.html)
- [std::ffi module](https://doc.rust-lang.org/std/ffi/index.html)
- [Miri — undefined behavior detector](https://github.com/rust-lang/miri)

---

**Previous**: [Modules and Cargo](./16_Modules_and_Cargo.md) | **Next**: [Project: CLI Tool](./18_Project_CLI_Tool.md)
