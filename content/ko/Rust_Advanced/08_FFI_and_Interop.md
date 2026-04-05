# 24. FFI와 상호운용

**이전**: [고급 비동기](./07_Advanced_Async.md) | **다음**: [WebAssembly](./09_WebAssembly.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `extern`을 사용하여 Rust에서 C 함수 호출 및 C에 Rust 함수 노출하기
2. 원시 포인터, C 문자열, C 호환 타입을 안전하게 다루기
3. `bindgen`으로 C 헤더에서 Rust 바인딩 자동 생성하기
4. `cbindgen`으로 Rust 코드에서 C 헤더 생성하기
5. PyO3로 Rust-Python 상호운용을 위한 Python 확장 빌드하기

---

Rust의 제로 비용 추상화와 메모리 안전성은 기존 코드베이스와 상호운용하는 성능 중심 컴포넌트 작성에 탁월한 언어입니다. 이 레슨은 저수준 C 상호운용부터 PyO3를 활용한 고수준 Python 바인딩까지 외부 함수 인터페이스(FFI, Foreign Function Interface)를 다룹니다.

## 목차
1. [FFI 기초](#1-ffi-기초)
2. [Rust에서 C 호출하기](#2-rust에서-c-호출하기)
3. [C 호환 타입](#3-c-호환-타입)
4. [C에 Rust 노출하기](#4-c에-rust-노출하기)
5. [C 문자열 다루기](#5-c-문자열-다루기)
6. [콜백과 함수 포인터](#6-콜백과-함수-포인터)
7. [bindgen: 바인딩 자동 생성](#7-bindgen-바인딩-자동-생성)
8. [cbindgen: C 헤더 생성](#8-cbindgen-c-헤더-생성)
9. [PyO3: Python을 위한 Rust](#9-pyo3-python을-위한-rust)
10. [FFI 안전 패턴](#10-ffi-안전-패턴)
11. [빌드와 링킹](#11-빌드와-링킹)
12. [연습문제](#12-연습문제)

---

## 1. FFI 기초

FFI는 Rust가 다른 언어로 작성된 함수를 호출하거나 그 반대를 가능하게 합니다. 다리는 **C ABI**(Application Binary Interface)입니다 — 거의 모든 언어가 지원하는 호출 규약입니다.

```
┌─────────────┐     C ABI      ┌──────────────┐
│  Rust 코드  │ ◄────────────► │   C 라이브러리│
└─────────────┘                └──────────────┘

┌─────────────┐     C ABI      ┌──────────────┐
│  Rust 코드  │ ◄────────────► │ Python (PyO3)│
└─────────────┘                └──────────────┘
```

핵심 개념:
- `extern "C"` — C 호출 규약 사용
- `#[repr(C)]` — C처럼 구조체 필드 배치
- `unsafe` — 모든 FFI 호출에 필요 (외부 코드를 검증할 수 없음)
- `#[no_mangle]` — Rust의 이름 맹글링(name mangling) 방지

---

## 2. Rust에서 C 호출하기

### 외부 함수 선언

```rust
use std::os::raw::{c_int, c_double, c_char};

// Rust가 호출할 C 함수 선언
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

### C 라이브러리에 링킹

```toml
# Cargo.toml — 시스템 라이브러리에 링킹
[build-dependencies]
cc = "1"  # C 소스 파일 컴파일용

# 또는 설치된 라이브러리에 링킹:
# [package.metadata.system-deps]
# openssl = "1"
```

빌드 스크립트 (`build.rs`):

```rust
// build.rs
fn main() {
    // C 파일을 컴파일하고 링킹
    cc::Build::new()
        .file("src/math_helper.c")
        .compile("math_helper");

    // 또는 기존 라이브러리에 링킹
    // println!("cargo:rustc-link-lib=ssl");
    // println!("cargo:rustc-link-search=/usr/local/lib");
}
```

C 소스 파일:

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

Rust 바인딩:

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

## 3. C 호환 타입

### #[repr(C)] 구조체

```rust
use std::os::raw::{c_int, c_float, c_char};

// C 구조체와 동일한 메모리 레이아웃
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

// C 호환 열거형
#[repr(C)]
enum Color {
    Red = 0,
    Green = 1,
    Blue = 2,
}

// 명시적 정수 표현이 있는 열거형
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

### 기본 타입 매핑

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
| `()` | `void` (반환) | `c_void` |

---

## 4. C에 Rust 노출하기

### #[no_mangle]과 extern "C"

```rust
use std::os::raw::c_int;

/// C에서 호출 가능한 Rust 함수
#[no_mangle]
pub extern "C" fn rust_add(a: c_int, b: c_int) -> c_int {
    a + b
}

/// 더 복잡한 함수 노출
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

### 불투명 타입 (핸들 패턴)

Rust 타입을 C에 불투명 포인터로 노출:

```rust
use std::os::raw::c_char;
use std::ffi::{CStr, CString};

pub struct Database {
    path: String,
    records: Vec<String>,
}

/// 새 데이터베이스 생성 — 불투명 포인터 반환
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

/// 레코드 삽입
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

/// 레코드 수 반환
#[no_mangle]
pub extern "C" fn db_count(db: *const Database) -> c_int {
    let db = unsafe {
        assert!(!db.is_null());
        &*db
    };
    db.records.len() as c_int
}

/// 데이터베이스 해제 — 메모리 누수 방지를 위해 반드시 호출
#[no_mangle]
pub extern "C" fn db_free(db: *mut Database) {
    if !db.is_null() {
        unsafe {
            drop(Box::from_raw(db));
        }
    }
}
```

C에서 사용:

```c
// database.h (cbindgen으로 생성)
typedef struct Database Database;

Database* db_create(const char* path);
int db_insert(Database* db, const char* record);
int db_count(const Database* db);
void db_free(Database* db);

// main.c
int main() {
    Database* db = db_create("test.db");
    db_insert(db, "레코드 1");
    db_insert(db, "레코드 2");
    printf("Count: %d\n", db_count(db));  // 2
    db_free(db);  // Rust가 할당한 메모리 해제
    return 0;
}
```

---

## 5. C 문자열 다루기

### CStr과 CString

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// CString — 소유권 있는 null 종료 문자열 (Rust → C)
fn rust_to_c() {
    let rust_string = "안녕하세요, C 세계!";
    let c_string = CString::new(rust_string).expect("CString::new 실패");

    // C에 전달할 포인터 얻기
    let ptr: *const c_char = c_string.as_ptr();

    // 중요: ptr을 사용하는 동안 c_string이 살아 있어야 함!
    unsafe {
        let len = libc::strlen(ptr);
        println!("C가 보는 문자열 길이: {len}");
    }
}

// CStr — 빌린 null 종료 문자열 (C → Rust)
unsafe fn c_to_rust(ptr: *const c_char) -> String {
    assert!(!ptr.is_null());
    let c_str = CStr::from_ptr(ptr);

    // &str로 변환 (유효한 UTF-8이면 빌림, 제로 카피)
    match c_str.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => c_str.to_string_lossy().into_owned(),
    }
}

// 공통 패턴: 문자열을 반환하는 C 함수 래핑
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

### OsStr과 OsString

플랫폼별 문자열(예: Windows의 파일 경로)을 위해:

```rust
use std::ffi::{OsStr, OsString};
use std::path::Path;

fn handle_path(path: &OsStr) {
    // OsStr은 일부 플랫폼에서 유효한 UTF-8이 아닐 수 있음
    match path.to_str() {
        Some(s) => println!("경로 (UTF-8): {s}"),
        None => println!("경로 (비UTF-8): {:?}", path),
    }
}
```

---

## 6. 콜백과 함수 포인터

### C에서 Rust 콜백 호출

```rust
use std::os::raw::c_int;

// C 호환 콜백을 받는 Rust 함수
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

// 콜백 함수
extern "C" fn double(x: c_int) -> c_int { x * 2 }

fn main() {
    let data = vec![1, 2, 3, 4, 5];
    let result = unsafe {
        apply_to_array(data.as_ptr(), data.len() as c_int, double)
    };
    println!("{result:?}");  // [2, 4, 6, 8, 10]
}
```

### 트레이트 객체를 통한 클로저를 콜백으로

C는 Rust 클로저(`extern "C"`가 아님)를 직접 호출할 수 없지만, 트램폴린(trampoline)을 사용할 수 있습니다:

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

## 7. bindgen: 바인딩 자동 생성

`bindgen`은 C/C++ 헤더를 읽고 Rust 바인딩을 자동으로 생성합니다:

```bash
cargo install bindgen-cli

# 헤더 파일에서 바인딩 생성
bindgen wrapper.h -o src/bindings.rs
```

### build.rs에서 bindgen 사용

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
    // cargo에 C 라이브러리 링킹 지시
    println!("cargo:rustc-link-lib=mylib");
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("바인딩 생성 불가");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("바인딩 저장 실패!");
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
        // 생성된 바인딩 사용
    }
}
```

### 예제: zlib 바인딩

```c
// wrapper.h
#include <zlib.h>
```

```rust
// bindgen이 바인딩을 생성한 후:
use std::ffi::CString;

fn main() {
    unsafe {
        let version = CStr::from_ptr(zlibVersion());
        println!("zlib 버전: {}", version.to_str().unwrap());

        // 데이터 압축
        let input = b"Rust에서 zlib 안녕하세요!";
        let mut output = vec![0u8; 1024];
        let mut output_len = output.len() as u64;

        let result = compress(
            output.as_mut_ptr(),
            &mut output_len,
            input.as_ptr(),
            input.len() as u64,
        );

        if result == Z_OK as i32 {
            println!("{}바이트를 {}바이트로 압축",
                input.len(), output_len);
        }
    }
}
```

---

## 8. cbindgen: C 헤더 생성

`cbindgen`은 Rust 소스를 읽고 C/C++ 헤더를 생성합니다:

```bash
cargo install cbindgen
```

```toml
# cbindgen.toml
language = "C"
include_guard = "MY_LIBRARY_H"
autogen_warning = "/* cbindgen에 의해 자동 생성됨. 수정하지 마세요. */"

[export]
include = ["Point", "Rect", "Color"]

[fn]
rename_args = "CamelCase"
```

```bash
cbindgen --config cbindgen.toml --crate my_library --output my_library.h
```

생성된 헤더:

```c
/* cbindgen에 의해 자동 생성됨. 수정하지 마세요. */

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

## 9. PyO3: Python을 위한 Rust

PyO3는 최소한의 보일러플레이트로 Rust로 Python 모듈을 작성할 수 있게 합니다:

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

### 기본 Python 모듈

```rust
use pyo3::prelude::*;

/// Python에서 호출 가능한 간단한 함수
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> String {
    (a + b).to_string()
}

/// Rust의 피보나치 (Python보다 훨씬 빠름)
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

/// Python 모듈 정의
#[pymodule]
fn my_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci, m)?)?;
    Ok(())
}
```

### Python 클래스

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

    // 정적 메서드
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

Python에서 사용:

```python
from geometry import Point

p1 = Point(3.0, 4.0)
p2 = Point.origin()
print(f"거리: {p1.distance(p2)}")  # 5.0
print(repr(p1))  # Point(3.0, 4.0)
```

### 오류 처리

```rust
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};

#[pyfunction]
fn parse_number(s: &str) -> PyResult<i64> {
    s.parse::<i64>().map_err(|e| {
        PyValueError::new_err(format!("'{s}' 파싱 불가: {e}"))
    })
}

#[pyfunction]
fn read_config(path: &str) -> PyResult<String> {
    std::fs::read_to_string(path).map_err(|e| {
        PyIOError::new_err(format!("'{path}' 읽기 불가: {e}"))
    })
}
```

### maturin으로 빌드

```bash
pip install maturin

# 현재 가상환경에 빌드 및 설치
maturin develop

# 배포용 휠 빌드
maturin build --release

# PyPI에 게시
maturin publish
```

---

## 10. FFI 안전 패턴

### 래퍼 타입

```rust
use std::os::raw::c_void;

// 어떤 라이브러리의 원시 C 핸들
extern "C" {
    fn create_handle() -> *mut c_void;
    fn destroy_handle(h: *mut c_void);
    fn handle_operation(h: *mut c_void, data: i32) -> i32;
}

// 안전한 Rust 래퍼
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

// 핸들이 스레드 간에 실수로 공유되지 않도록 보장
// (C 라이브러리가 스레드 안전하지 않은 경우)
// impl !Send for SafeHandle {}
// impl !Sync for SafeHandle {}
```

### 경계에서 입력 검증

```rust
use std::ffi::CStr;
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn process_data(
    data: *const u8,
    len: usize,
    name: *const c_char,
) -> i32 {
    // FFI 경계에서 모든 포인터 검증
    if data.is_null() || name.is_null() {
        return -1;  // 오류 코드
    }

    // 가능한 빨리 안전한 Rust 타입으로 변환
    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return -2,  // 유효하지 않은 UTF-8
        }
    };

    // 이제 안전한 Rust 타입으로 작업
    println!("'{name_str}'을 위해 {} 바이트 처리 중", data_slice.len());
    0  // 성공
}
```

---

## 11. 빌드와 링킹

### 정적 vs 동적 링킹

```rust
// build.rs
fn main() {
    // 정적 링킹 — 라이브러리가 바이너리에 포함됨
    println!("cargo:rustc-link-lib=static=mylib");

    // 동적 링킹 — 런타임에 라이브러리 로드
    println!("cargo:rustc-link-lib=dylib=mylib");

    // 시스템 라이브러리 (OS가 정적 또는 동적 결정)
    println!("cargo:rustc-link-lib=mylib");

    // 라이브러리 검색 경로
    println!("cargo:rustc-link-search=native=/usr/local/lib");
}
```

### 라이브러리의 크레이트 타입

```toml
[lib]
crate-type = ["cdylib"]    # C/Python용 동적 라이브러리 (.so, .dylib, .dll)
# crate-type = ["staticlib"] # C용 정적 라이브러리 (.a, .lib)
# crate-type = ["rlib"]      # Rust 라이브러리 (기본값)
```

### 크로스 플랫폼 고려사항

```rust
// 플랫폼별 FFI를 위한 조건부 컴파일
#[cfg(target_os = "linux")]
extern "C" {
    fn epoll_create1(flags: i32) -> i32;
}

#[cfg(target_os = "macos")]
extern "C" {
    fn kqueue() -> i32;
}

// 아키텍처별
#[cfg(target_arch = "x86_64")]
extern "C" {
    fn _mm_pause();  // x86 인트린식
}
```

---

## 12. 연습문제

1. **C 수학 라이브러리 래퍼**: `libm` 함수들(`sin`, `cos`, `exp`, `log`)의 안전한 Rust 래퍼를 작성하세요. 래퍼는 공개 API에 `unsafe`가 없이 네이티브 Rust `f64` 값을 받고 반환해야 합니다.

2. **불투명 타입 라이브러리**: 불투명 포인터를 통해 C에 `StringBuffer` 타입을 노출하는 Rust 라이브러리를 만드세요. `create`, `append`, `get_str`, `length`, `free` 연산을 지원하세요.

3. **bindgen 실습**: 구조체, 열거형, 함수가 있는 간단한 C 헤더에 대한 Rust 바인딩을 bindgen으로 생성하세요. 생성된 바인딩 주위에 안전한 래퍼 타입을 작성하세요.

4. **PyO3 데이터 처리기**: CSV 데이터 처리를 위한 `DataFrame`류 클래스를 제공하는 Python 모듈을 Rust로 작성하세요. `from_csv`, `filter`, `sort`, `to_json` 메서드를 구현하세요.

5. **콜백 브릿지**: 정렬 비교자 콜백을 받는 C 라이브러리를 구현하세요. 트램폴린 패턴을 사용하여 클로저를 콜백으로 전달하는 Rust 코드를 작성하세요.

---

## 참고 자료

- [The Rustonomicon: FFI](https://doc.rust-lang.org/nomicon/ffi.html)
- [bindgen User Guide](https://rust-lang.github.io/rust-bindgen/)
- [cbindgen User Guide](https://github.com/mozilla/cbindgen)
- [PyO3 User Guide](https://pyo3.rs/)
- [maturin documentation](https://www.maturin.rs/)

---

**이전**: [고급 비동기](./07_Advanced_Async.md) | **다음**: [WebAssembly](./09_WebAssembly.md)
