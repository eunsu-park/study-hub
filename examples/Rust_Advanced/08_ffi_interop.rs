// 08_ffi_interop.rs — FFI patterns: calling C, exposing Rust, C strings
//
// Run: rustc 08_ffi_interop.rs && ./08_ffi_interop

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

fn main() {
    println!("=== Calling C Standard Library ===");
    call_c_stdlib();

    println!("\n=== C String Handling ===");
    c_string_demo();

    println!("\n=== Exposing Rust to C ===");
    rust_to_c_demo();

    println!("\n=== Opaque Pointer Pattern ===");
    opaque_pointer_demo();
}

// --- Calling C functions ---

extern "C" {
    fn abs(input: i32) -> i32;
    fn strlen(s: *const c_char) -> usize;
    fn atoi(s: *const c_char) -> i32;
}

fn call_c_stdlib() {
    unsafe {
        println!("  abs(-42) = {}", abs(-42));
        println!("  abs(10) = {}", abs(10));

        let c_str = CString::new("hello").unwrap();
        println!("  strlen(\"hello\") = {}", strlen(c_str.as_ptr()));

        let num_str = CString::new("12345").unwrap();
        println!("  atoi(\"12345\") = {}", atoi(num_str.as_ptr()));
    }
}

// --- C string conversions ---

fn c_string_demo() {
    // Rust String → CString (adds null terminator)
    let rust_str = "Hello from Rust";
    let c_string = CString::new(rust_str).expect("CString::new failed");
    println!("  Rust → CString: {:?}", c_string);
    println!("  As bytes (with null): {:?}", c_string.as_bytes_with_nul());

    // CString → *const c_char (for passing to C)
    let ptr: *const c_char = c_string.as_ptr();
    println!("  Pointer: {ptr:?}");

    // *const c_char → &CStr → &str (for receiving from C)
    unsafe {
        let c_str: &CStr = CStr::from_ptr(ptr);
        let str_slice: &str = c_str.to_str().expect("Invalid UTF-8");
        println!("  CStr → &str: \"{str_slice}\"");
    }

    // Strings with interior null bytes — CString::new will error
    match CString::new("hello\0world") {
        Ok(_) => println!("  Unexpected success"),
        Err(e) => println!("  Interior null at position: {}", e.nul_position()),
    }
}

// --- Exposing Rust functions with C ABI ---

// #[no_mangle] prevents name mangling so C can find the symbol
// extern "C" specifies the C calling convention
#[no_mangle]
pub extern "C" fn rust_add(a: i32, b: i32) -> i32 {
    a + b
}

#[no_mangle]
pub extern "C" fn rust_multiply(a: f64, b: f64) -> f64 {
    a * b
}

// Returning a C string (caller must free with rust_free_string)
#[no_mangle]
pub extern "C" fn rust_greeting(name: *const c_char) -> *mut c_char {
    let c_str = unsafe { CStr::from_ptr(name) };
    let name = c_str.to_str().unwrap_or("unknown");
    let greeting = format!("Hello, {name} from Rust!");
    CString::new(greeting).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn rust_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)); }
    }
}

fn rust_to_c_demo() {
    // Call our own extern "C" functions
    println!("  rust_add(3, 4) = {}", rust_add(3, 4));
    println!("  rust_multiply(2.5, 4.0) = {}", rust_multiply(2.5, 4.0));

    let name = CString::new("World").unwrap();
    let greeting_ptr = rust_greeting(name.as_ptr());
    let greeting = unsafe { CStr::from_ptr(greeting_ptr) };
    println!("  rust_greeting: {}", greeting.to_str().unwrap());
    rust_free_string(greeting_ptr); // Clean up
}

// --- Opaque pointer pattern ---

// Expose a Rust struct through an opaque handle
struct Counter {
    value: i64,
    label: String,
}

type CounterHandle = *mut Counter;

#[no_mangle]
pub extern "C" fn counter_new(label: *const c_char) -> CounterHandle {
    let label = unsafe { CStr::from_ptr(label) }
        .to_str()
        .unwrap_or("default")
        .to_string();
    Box::into_raw(Box::new(Counter { value: 0, label }))
}

#[no_mangle]
pub extern "C" fn counter_increment(handle: CounterHandle) {
    if let Some(counter) = unsafe { handle.as_mut() } {
        counter.value += 1;
    }
}

#[no_mangle]
pub extern "C" fn counter_get(handle: CounterHandle) -> i64 {
    unsafe { handle.as_ref() }.map(|c| c.value).unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn counter_free(handle: CounterHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)); }
    }
}

fn opaque_pointer_demo() {
    let label = CString::new("requests").unwrap();
    let counter = counter_new(label.as_ptr());

    counter_increment(counter);
    counter_increment(counter);
    counter_increment(counter);

    println!("  Counter value: {}", counter_get(counter));

    counter_free(counter);
    println!("  Counter freed successfully");
}
