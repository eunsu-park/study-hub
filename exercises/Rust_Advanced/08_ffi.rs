// Exercise: FFI and Interop
// Practice calling C from Rust and exposing Rust to C.
//
// Run: rustc 24_ffi.rs && ./24_ffi

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_double};

// Exercise 1: Call C math functions safely
extern "C" {
    fn abs(input: c_int) -> c_int;
    fn sqrt(input: c_double) -> c_double;
    fn pow(base: c_double, exp: c_double) -> c_double;
}

fn safe_abs(n: i32) -> i32 {
    unsafe { abs(n) }
}

fn safe_sqrt(n: f64) -> Option<f64> {
    if n < 0.0 { return None; }
    Some(unsafe { sqrt(n) })
}

fn safe_pow(base: f64, exp: f64) -> f64 {
    unsafe { pow(base, exp) }
}

// Exercise 2: C string conversion
fn c_string_roundtrip(s: &str) -> String {
    let c_str = CString::new(s).expect("CString::new failed");
    let ptr = c_str.as_ptr();
    unsafe {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

// Exercise 3: Expose Rust function with C ABI
#[no_mangle]
pub extern "C" fn rust_fibonacci(n: c_int) -> c_int {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let (mut a, mut b) = (0i32, 1i32);
            for _ in 2..=n {
                let t = a + b;
                a = b;
                b = t;
            }
            b
        }
    }
}

// Exercise 4: Safe wrapper around opaque handle
struct SafeBuffer {
    data: Vec<u8>,
}

impl SafeBuffer {
    fn new(capacity: usize) -> Self {
        SafeBuffer { data: Vec::with_capacity(capacity) }
    }

    fn write(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }

    fn as_slice(&self) -> &[u8] {
        &self.data
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

fn main() {
    // Test Exercise 1
    println!("abs(-42) = {}", safe_abs(-42));
    println!("sqrt(144) = {:?}", safe_sqrt(144.0));
    println!("sqrt(-1) = {:?}", safe_sqrt(-1.0));
    println!("pow(2, 10) = {}", safe_pow(2.0, 10.0));

    // Test Exercise 2
    let original = "Hello, FFI!";
    let roundtrip = c_string_roundtrip(original);
    assert_eq!(original, roundtrip);
    println!("C string roundtrip: '{original}' -> '{roundtrip}'");

    // Test Exercise 3
    println!("fibonacci(10) = {}", rust_fibonacci(10));
    println!("fibonacci(20) = {}", rust_fibonacci(20));

    // Test Exercise 4
    let mut buf = SafeBuffer::new(64);
    buf.write(b"Hello, ");
    buf.write(b"world!");
    assert_eq!(buf.len(), 13);
    assert_eq!(buf.as_slice(), b"Hello, world!");
    println!("Buffer: {:?}", std::str::from_utf8(buf.as_slice()).unwrap());

    println!("\nAll exercises passed!");
}
