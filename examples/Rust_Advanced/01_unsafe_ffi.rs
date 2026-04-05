// 17_unsafe_ffi.rs — Unsafe Rust and FFI basics
//
// Run: rustc 17_unsafe_ffi.rs && ./17_unsafe_ffi

fn main() {
    println!("=== Raw Pointers ===");
    raw_pointers();

    println!("\n=== Safe Abstraction over Unsafe ===");
    safe_abstraction();

    println!("\n=== Calling C Functions (FFI) ===");
    ffi_demo();

    println!("\n=== Mutable Statics ===");
    static_demo();
}

fn raw_pointers() {
    let mut x = 42;

    // Creating raw pointers is safe — dereferencing is unsafe
    let r1 = &x as *const i32; // Immutable raw pointer
    let r2 = &mut x as *mut i32; // Mutable raw pointer

    unsafe {
        println!("r1 = {}", *r1);
        println!("r2 = {}", *r2);

        // Modify through raw pointer
        *r2 = 100;
        println!("After modification: {}", *r1);
    }

    // Raw pointers can point to arbitrary addresses (dangerous!)
    // let address = 0x012345usize;
    // let _r = address as *const i32; // Valid to create, unsafe to deref
}

// Creating a safe abstraction over unsafe code
fn split_at_mut(values: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = values.len();
    assert!(mid <= len, "mid out of bounds");

    let ptr = values.as_mut_ptr();

    // The borrow checker can't verify that these two slices don't overlap,
    // but we know they don't because [0..mid] and [mid..len] are disjoint
    unsafe {
        (
            std::slice::from_raw_parts_mut(ptr, mid),
            std::slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

fn safe_abstraction() {
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let (left, right) = split_at_mut(&mut data, 3);
    println!("Left: {left:?}");
    println!("Right: {right:?}");

    left[0] = 99;
    right[0] = 88;
    println!("Modified left: {left:?}, right: {right:?}");
}

// FFI — calling C standard library functions
extern "C" {
    fn abs(input: i32) -> i32;
    fn sqrt(input: f64) -> f64;
}

fn ffi_demo() {
    unsafe {
        println!("abs(-42) = {}", abs(-42));
        println!("sqrt(144.0) = {}", sqrt(144.0));
    }
}

// Mutable static variables require unsafe
static mut REQUEST_COUNT: u32 = 0;

fn increment_count() {
    unsafe {
        REQUEST_COUNT += 1;
    }
}

fn static_demo() {
    increment_count();
    increment_count();
    increment_count();
    unsafe {
        println!("Request count: {REQUEST_COUNT}");
    }
    // Note: in real code, use AtomicU32 or Mutex instead of static mut
}
