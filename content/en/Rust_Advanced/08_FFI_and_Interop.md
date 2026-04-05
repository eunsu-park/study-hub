# 24. FFI and Interop

**Previous**: [Advanced Async](./07_Advanced_Async.md) | **Next**: [WebAssembly](./09_WebAssembly.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Call C functions from Rust and expose Rust functions to C using `extern`
2. Work with raw pointers, C strings, and C-compatible types safely
3. Use `bindgen` to auto-generate Rust bindings from C headers
4. Use `cbindgen` to generate C headers from Rust code
5. Build Python extensions with PyO3 for Rust-Python interop

---

Rust's zero-cost abstractions and memory safety make it an excellent language for writing performance-critical components that interoperate with existing codebases. This lesson covers the Foreign Function Interface (FFI), from low-level C interop to high-level Python bindings with PyO3.

## Table of Contents
1. [FFI Fundamentals](#1-ffi-fundamentals)
2. [Calling C from Rust](#2-calling-c-from-rust)
3. [C-Compatible Types](#3-c-compatible-types)
4. [Exposing Rust to C](#4-exposing-rust-to-c)
5. [Working with C Strings](#5-working-with-c-strings)
6. [Callbacks and Function Pointers](#6-callbacks-and-function-pointers)
7. [bindgen: Auto-Generating Bindings](#7-bindgen-auto-generating-bindings)
8. [cbindgen: Generating C Headers](#8-cbindgen-generating-c-headers)
9. [PyO3: Rust for Python](#9-pyo3-rust-for-python)
10. [Safety Patterns for FFI](#10-safety-patterns-for-ffi)
11. [Building and Linking](#11-building-and-linking)
12. [Exercises](#12-exercises)

---

## 1. FFI Fundamentals

FFI allows Rust to call functions written in other languages and vice versa. The bridge is the **C ABI** (Application Binary Interface) — the calling convention that nearly every language supports.

```
┌─────────────┐     C ABI      ┌──────────────┐
│  Rust Code  │ ◄────────────► │   C Library  │
└─────────────┘                └──────────────┘

┌─────────────┐     C ABI      ┌──────────────┐
│  Rust Code  │ ◄────────────► │ Python (PyO3)│
└─────────────┘                └──────────────┘
```

Key concepts:
- `extern "C"` — use the C calling convention
- `#[repr(C)]` — lay out struct fields like C would
- `unsafe` — required for all FFI calls (Rust can't verify foreign code)
- `#[no_mangle]` — prevent Rust from mangling the function name

---

## 2. Calling C from Rust

### Declaring External Functions

```rust
use std::os::raw::{c_int, c_double, c_char};

// Declare C functions that Rust will call
extern "C" {
    fn abs(input: c_int) -> c_int;
    fn sqrt(input: c_double) -> c_double;
    fn strlen(s: *const c_char) -> usize;
}

fn main() {
    unsafe {
        println!("abs(-5) = {}", abs(-5));
        println!("sqrt(2.0) = {}", sqrt(2.0));

        let s = b"Hello\0".as_ptr() as *const c_char;
        println!("strlen(\"Hello\") = {}", strlen(s));
    }
}
```

### Linking to a C Library

```toml
# Cargo.toml — link to system library
[build-dependencies]
cc = "1"  # For compiling C source files

# Or link to an installed library:
# [package.metadata.system-deps]
# openssl = "1"
```

Build script (`build.rs`):

```rust
// build.rs
fn main() {
    // Compile a C file and link it
    cc::Build::new()
        .file("src/math_helper.c")
        .compile("math_helper");

    // Or link to an existing library
    // println!("cargo:rustc-link-lib=ssl");
    // println!("cargo:rustc-link-search=/usr/local/lib");
}
```

The C source file:

```c
// src/math_helper.c
#include <math.h>

double hypotenuse(double a, double b) {
    return sqrt(a * a + b * b);
}

int fibonacci(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}
```

Rust bindings:

```rust
extern "C" {
    fn hypotenuse(a: f64, b: f64) -> f64;
    fn fibonacci(n: i32) -> i32;
}

fn main() {
    unsafe {
        println!("hypotenuse(3, 4) = {}", hypotenuse(3.0, 4.0));
        println!("fibonacci(10) = {}", fibonacci(10));
    }
}
```

---

## 3. C-Compatible Types

### #[repr(C)] Structs

```rust
use std::os::raw::{c_int, c_float, c_char};

// This struct has the same memory layout as a C struct
#[repr(C)]
struct Point {
    x: c_float,
    y: c_float,
}

#[repr(C)]
struct Rect {
    origin: Point,
    width: c_float,
    height: c_float,
}

// C-compatible enum
#[repr(C)]
enum Color {
    Red = 0,
    Green = 1,
    Blue = 2,
}

// Enum with explicit integer representation
#[repr(u8)]
enum Status {
    Active = 1,
    Inactive = 0,
    Error = 255,
}

extern "C" {
    fn draw_rect(rect: *const Rect, color: Color);
}
```

### Primitive Type Mapping

| Rust | C | `std::os::raw` |
|------|---|-----------------|
| `i8` | `int8_t` / `char` | `c_char` |
| `i16` | `int16_t` | `c_short` |
| `i32` | `int32_t` | `c_int` |
| `i64` | `int64_t` | `c_longlong` |
| `u8` | `uint8_t` | `c_uchar` |
| `u16` | `uint16_t` | `c_ushort` |
| `u32` | `uint32_t` | `c_uint` |
| `u64` | `uint64_t` | `c_ulonglong` |
| `f32` | `float` | `c_float` |
| `f64` | `double` | `c_double` |
| `bool` | `_Bool` | — |
| `*const T` | `const T*` | — |
| `*mut T` | `T*` | — |
| `()` | `void` (return) | `c_void` |

---

## 4. Exposing Rust to C

### #[no_mangle] and extern "C"

```rust
use std::os::raw::c_int;

/// A Rust function callable from C
#[no_mangle]
pub extern "C" fn rust_add(a: c_int, b: c_int) -> c_int {
    a + b
}

/// Expose a more complex function
#[no_mangle]
pub extern "C" fn rust_fibonacci(n: c_int) -> c_int {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let (mut a, mut b) = (0, 1);
            for _ in 2..=n {
                let temp = a + b;
                a = b;
                b = temp;
            }
            b
        }
    }
}
```

### Opaque Types (Handle Pattern)

Expose Rust types to C as opaque pointers:

```rust
use std::os::raw::c_char;
use std::ffi::{CStr, CString};

pub struct Database {
    path: String,
    records: Vec<String>,
}

/// Create a new database — returns opaque pointer
#[no_mangle]
pub extern "C" fn db_create(path: *const c_char) -> *mut Database {
    let path = unsafe {
        assert!(!path.is_null());
        CStr::from_ptr(path).to_string_lossy().into_owned()
    };

    let db = Database {
        path,
        records: Vec::new(),
    };

    Box::into_raw(Box::new(db))
}

/// Insert a record
#[no_mangle]
pub extern "C" fn db_insert(db: *mut Database, record: *const c_char) -> c_int {
    let db = unsafe {
        assert!(!db.is_null());
        &mut *db
    };

    let record = unsafe {
        assert!(!record.is_null());
        CStr::from_ptr(record).to_string_lossy().into_owned()
    };

    db.records.push(record);
    db.records.len() as c_int
}

/// Get record count
#[no_mangle]
pub extern "C" fn db_count(db: *const Database) -> c_int {
    let db = unsafe {
        assert!(!db.is_null());
        &*db
    };
    db.records.len() as c_int
}

/// Free the database — MUST be called to avoid memory leak
#[no_mangle]
pub extern "C" fn db_free(db: *mut Database) {
    if !db.is_null() {
        unsafe {
            drop(Box::from_raw(db));
        }
    }
}
```

C usage:

```c
// database.h (generate with cbindgen)
typedef struct Database Database;

Database* db_create(const char* path);
int db_insert(Database* db, const char* record);
int db_count(const Database* db);
void db_free(Database* db);

// main.c
int main() {
    Database* db = db_create("test.db");
    db_insert(db, "record 1");
    db_insert(db, "record 2");
    printf("Count: %d\n", db_count(db));  // 2
    db_free(db);  // Free Rust-allocated memory
    return 0;
}
```

---

## 5. Working with C Strings

### CStr and CString

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// CString — owned, null-terminated string (Rust → C)
fn rust_to_c() {
    let rust_string = "Hello, C world!";
    let c_string = CString::new(rust_string).expect("CString::new failed");

    // Get a pointer to pass to C
    let ptr: *const c_char = c_string.as_ptr();

    // IMPORTANT: c_string must live as long as ptr is used!
    unsafe {
        let len = libc::strlen(ptr);
        println!("C sees string of length {len}");
    }
}

// CStr — borrowed, null-terminated string (C → Rust)
unsafe fn c_to_rust(ptr: *const c_char) -> String {
    assert!(!ptr.is_null());
    let c_str = CStr::from_ptr(ptr);

    // To &str (borrowing, zero-copy if valid UTF-8)
    match c_str.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => c_str.to_string_lossy().into_owned(),
    }
}

// Common pattern: wrap C function that returns a string
extern "C" {
    fn getenv(name: *const c_char) -> *const c_char;
}

fn get_env_var(name: &str) -> Option<String> {
    let c_name = CString::new(name).ok()?;
    unsafe {
        let ptr = getenv(c_name.as_ptr());
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}

fn main() {
    if let Some(home) = get_env_var("HOME") {
        println!("HOME = {home}");
    }
}
```

### OsStr and OsString

For platform-specific strings (file paths on Windows, for example):

```rust
use std::ffi::{OsStr, OsString};
use std::path::Path;

fn handle_path(path: &OsStr) {
    // OsStr might not be valid UTF-8 on some platforms
    match path.to_str() {
        Some(s) => println!("Path (UTF-8): {s}"),
        None => println!("Path (non-UTF-8): {:?}", path),
    }
}
```

---

## 6. Callbacks and Function Pointers

### C Calling Rust Callbacks

```rust
use std::os::raw::c_int;

// Rust function that takes a C-compatible callback
#[no_mangle]
pub extern "C" fn apply_to_array(
    arr: *const c_int,
    len: c_int,
    callback: extern "C" fn(c_int) -> c_int,
) -> Vec<c_int> {
    let slice = unsafe {
        std::slice::from_raw_parts(arr, len as usize)
    };

    slice.iter().map(|&x| callback(x)).collect()
}

// A callback function
extern "C" fn double(x: c_int) -> c_int { x * 2 }

fn main() {
    let data = vec![1, 2, 3, 4, 5];
    let result = unsafe {
        apply_to_array(data.as_ptr(), data.len() as c_int, double)
    };
    println!("{result:?}");  // [2, 4, 6, 8, 10]
}
```

### Closures as Callbacks (via Trait Objects)

C can't directly call Rust closures (they're not `extern "C"`), but we can use a trampoline:

```rust
use std::os::raw::c_void;

type CCallback = extern "C" fn(*mut c_void, i32) -> i32;

extern "C" fn trampoline(data: *mut c_void, value: i32) -> i32 {
    let closure: &mut Box<dyn FnMut(i32) -> i32> = unsafe {
        &mut *(data as *mut Box<dyn FnMut(i32) -> i32>)
    };
    closure(value)
}

fn with_callback<F>(values: &[i32], mut f: F) -> Vec<i32>
where
    F: FnMut(i32) -> i32,
{
    let mut closure: Box<dyn FnMut(i32) -> i32> = Box::new(f);
    let data = &mut closure as *mut Box<dyn FnMut(i32) -> i32> as *mut c_void;

    values.iter().map(|&v| trampoline(data, v)).collect()
}

fn main() {
    let multiplier = 3;
    let result = with_callback(&[1, 2, 3, 4], |x| x * multiplier);
    println!("{result:?}");  // [3, 6, 9, 12]
}
```

---

## 7. bindgen: Auto-Generating Bindings

`bindgen` reads C/C++ headers and generates Rust bindings automatically:

```bash
cargo install bindgen-cli

# Generate bindings from a header file
bindgen wrapper.h -o src/bindings.rs
```

### Using bindgen in build.rs

```toml
# Cargo.toml
[build-dependencies]
bindgen = "0.70"
```

```rust
// build.rs
use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to link the C library
    println!("cargo:rustc-link-lib=mylib");
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
```

```rust
// src/main.rs
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

fn main() {
    unsafe {
        // Use the generated bindings
    }
}
```

### Example: Binding to zlib

```c
// wrapper.h
#include <zlib.h>
```

```rust
// After bindgen generates bindings:
use std::ffi::CString;

fn main() {
    unsafe {
        let version = CStr::from_ptr(zlibVersion());
        println!("zlib version: {}", version.to_str().unwrap());

        // Compress data
        let input = b"Hello, zlib from Rust!";
        let mut output = vec![0u8; 1024];
        let mut output_len = output.len() as u64;

        let result = compress(
            output.as_mut_ptr(),
            &mut output_len,
            input.as_ptr(),
            input.len() as u64,
        );

        if result == Z_OK as i32 {
            println!("Compressed {} bytes to {} bytes",
                input.len(), output_len);
        }
    }
}
```

---

## 8. cbindgen: Generating C Headers

`cbindgen` reads Rust source and generates C/C++ headers:

```bash
cargo install cbindgen
```

```toml
# cbindgen.toml
language = "C"
include_guard = "MY_LIBRARY_H"
autogen_warning = "/* Auto-generated by cbindgen. Do not edit. */"

[export]
include = ["Point", "Rect", "Color"]

[fn]
rename_args = "CamelCase"
```

```bash
cbindgen --config cbindgen.toml --crate my_library --output my_library.h
```

Generated header:

```c
/* Auto-generated by cbindgen. Do not edit. */

#ifndef MY_LIBRARY_H
#define MY_LIBRARY_H

#include <stdint.h>

typedef struct Point {
    float x;
    float y;
} Point;

typedef struct Rect {
    Point origin;
    float width;
    float height;
} Rect;

int32_t rust_add(int32_t A, int32_t B);
int32_t rust_fibonacci(int32_t N);

#endif /* MY_LIBRARY_H */
```

---

## 9. PyO3: Rust for Python

PyO3 enables writing Python modules in Rust with minimal boilerplate:

```toml
# Cargo.toml
[package]
name = "my-python-module"
version = "0.1.0"
edition = "2021"

[lib]
name = "my_module"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
```

### Basic Python Module

```rust
use pyo3::prelude::*;

/// A simple function callable from Python
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> String {
    (a + b).to_string()
}

/// Fibonacci in Rust (much faster than Python)
#[pyfunction]
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let (mut a, mut b) = (0u64, 1u64);
            for _ in 2..=n {
                let temp = a + b;
                a = b;
                b = temp;
            }
            b
        }
    }
}

/// Python module definition
#[pymodule]
fn my_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci, m)?)?;
    Ok(())
}
```

### Python Classes

```rust
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
struct Point {
    #[pyo3(get, set)]
    x: f64,
    #[pyo3(get, set)]
    y: f64,
}

#[pymethods]
impl Point {
    #[new]
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    fn __repr__(&self) -> String {
        format!("Point({}, {})", self.x, self.y)
    }

    fn __str__(&self) -> String {
        format!("({}, {})", self.x, self.y)
    }

    // Static method
    #[staticmethod]
    fn origin() -> Self {
        Point { x: 0.0, y: 0.0 }
    }
}

#[pymodule]
fn geometry(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point>()?;
    Ok(())
}
```

Python usage:

```python
from geometry import Point

p1 = Point(3.0, 4.0)
p2 = Point.origin()
print(f"Distance: {p1.distance(p2)}")  # 5.0
print(repr(p1))  # Point(3.0, 4.0)
```

### Error Handling

```rust
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};

#[pyfunction]
fn parse_number(s: &str) -> PyResult<i64> {
    s.parse::<i64>().map_err(|e| {
        PyValueError::new_err(format!("Cannot parse '{s}': {e}"))
    })
}

#[pyfunction]
fn read_config(path: &str) -> PyResult<String> {
    std::fs::read_to_string(path).map_err(|e| {
        PyIOError::new_err(format!("Cannot read '{path}': {e}"))
    })
}
```

### Building with maturin

```bash
pip install maturin

# Build and install in the current virtualenv
maturin develop

# Build a wheel for distribution
maturin build --release

# Publish to PyPI
maturin publish
```

---

## 10. Safety Patterns for FFI

### Wrapper Types

```rust
use std::os::raw::c_void;

// Raw C handle from some library
extern "C" {
    fn create_handle() -> *mut c_void;
    fn destroy_handle(h: *mut c_void);
    fn handle_operation(h: *mut c_void, data: i32) -> i32;
}

// Safe Rust wrapper
pub struct SafeHandle {
    raw: *mut c_void,
}

impl SafeHandle {
    pub fn new() -> Option<Self> {
        let raw = unsafe { create_handle() };
        if raw.is_null() {
            None
        } else {
            Some(SafeHandle { raw })
        }
    }

    pub fn operate(&self, data: i32) -> i32 {
        unsafe { handle_operation(self.raw, data) }
    }
}

impl Drop for SafeHandle {
    fn drop(&mut self) {
        unsafe {
            destroy_handle(self.raw);
        }
    }
}

// Ensure the handle is not accidentally shared across threads
// (unless the C library is thread-safe)
// impl !Send for SafeHandle {}
// impl !Sync for SafeHandle {}
```

### Validating Input at the Boundary

```rust
use std::ffi::CStr;
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn process_data(
    data: *const u8,
    len: usize,
    name: *const c_char,
) -> i32 {
    // Validate all pointers at the FFI boundary
    if data.is_null() || name.is_null() {
        return -1;  // Error code
    }

    // Convert to safe Rust types ASAP
    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return -2,  // Invalid UTF-8
        }
    };

    // Now work with safe Rust types
    println!("Processing {} bytes for '{name_str}'", data_slice.len());
    0  // Success
}
```

---

## 11. Building and Linking

### Static vs Dynamic Linking

```rust
// build.rs
fn main() {
    // Static linking — library is embedded in the binary
    println!("cargo:rustc-link-lib=static=mylib");

    // Dynamic linking — library loaded at runtime
    println!("cargo:rustc-link-lib=dylib=mylib");

    // System library (OS decides static or dynamic)
    println!("cargo:rustc-link-lib=mylib");

    // Search path for libraries
    println!("cargo:rustc-link-search=native=/usr/local/lib");
}
```

### Crate Type for Libraries

```toml
[lib]
crate-type = ["cdylib"]    # Dynamic library for C/Python (.so, .dylib, .dll)
# crate-type = ["staticlib"] # Static library for C (.a, .lib)
# crate-type = ["rlib"]      # Rust library (default)
```

### Cross-Platform Considerations

```rust
// Conditional compilation for platform-specific FFI
#[cfg(target_os = "linux")]
extern "C" {
    fn epoll_create1(flags: i32) -> i32;
}

#[cfg(target_os = "macos")]
extern "C" {
    fn kqueue() -> i32;
}

// Architecture-specific
#[cfg(target_arch = "x86_64")]
extern "C" {
    fn _mm_pause();  // x86 intrinsic
}
```

---

## 12. Exercises

1. **C math library wrapper**: Write a safe Rust wrapper around `libm` functions (`sin`, `cos`, `exp`, `log`). The wrapper should take and return native Rust `f64` values with no `unsafe` in the public API.

2. **Opaque type library**: Create a Rust library that exposes a `StringBuffer` type to C via opaque pointers. Support `create`, `append`, `get_str`, `length`, and `free` operations.

3. **bindgen practice**: Use bindgen to generate Rust bindings for a simple C header with structs, enums, and functions. Write safe wrapper types around the generated bindings.

4. **PyO3 data processor**: Write a Python module in Rust that provides a `DataFrame`-like class for processing CSV data. Implement `from_csv`, `filter`, `sort`, and `to_json` methods.

5. **Callback bridge**: Implement a C library that takes a sorting comparator callback. Write Rust code that passes a closure as the callback using the trampoline pattern.

---

## References

- [The Rustonomicon: FFI](https://doc.rust-lang.org/nomicon/ffi.html)
- [bindgen User Guide](https://rust-lang.github.io/rust-bindgen/)
- [cbindgen User Guide](https://github.com/mozilla/cbindgen)
- [PyO3 User Guide](https://pyo3.rs/)
- [maturin documentation](https://www.maturin.rs/)

---

**Previous**: [Advanced Async](./07_Advanced_Async.md) | **Next**: [WebAssembly](./09_WebAssembly.md)
