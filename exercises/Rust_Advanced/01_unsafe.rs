// Exercise: Unsafe Rust
// Practice with raw pointers, unsafe blocks, extern functions, and unsafe traits.
//
// Run: rustc 17_unsafe.rs && ./17_unsafe

use std::slice;

// ============================================================
// Exercise 1: Raw Pointer Basics
// ============================================================
// Create raw pointers from references and dereference them safely.

fn raw_pointer_sum(a: &i32, b: &i32) -> i32 {
    // TODO: Create a raw const pointer to each argument using `as *const i32`.
    // Dereference both inside an `unsafe` block and return their sum.
    todo!()
}

fn swap_via_raw(a: &mut i32, b: &mut i32) {
    // TODO: Create *mut i32 raw pointers to a and b.
    // Inside an `unsafe` block, swap their values using the pointers.
    // Hint: read the value through one pointer before overwriting.
    todo!()
}

// ============================================================
// Exercise 2: Safe Wrapper Around Unsafe Code
// ============================================================
// Implement a safe split_at_mut equivalent using raw pointers.
// The standard library uses exactly this pattern internally.

fn split_at_mut_custom(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    // TODO: Verify `mid <= slice.len()` and panic with a clear message if not.
    // Then use `slice.as_mut_ptr()` to get a *mut i32 raw pointer.
    // Inside `unsafe`:
    //   - Return (slice::from_raw_parts_mut(ptr, mid),
    //             slice::from_raw_parts_mut(ptr.add(mid), slice.len() - mid))
    // This is safe because the two sub-slices never overlap.
    todo!()
}

// ============================================================
// Exercise 3: Extern "C" Functions
// ============================================================
// Declare and call a C standard library function via FFI.

extern "C" {
    // Declaration of C's abs() from <stdlib.h>
    fn abs(n: i32) -> i32;
}

fn safe_abs(n: i32) -> i32 {
    // TODO: Call the extern `abs` function inside an `unsafe` block.
    // Return the result.
    todo!()
}

// ============================================================
// Exercise 4: Unsafe Trait
// ============================================================
// Define an unsafe trait and implement it for a custom type.
// `unsafe trait` signals that implementors must uphold invariants
// the compiler cannot verify (similar to Send/Sync).

/// Safety: implementors must ensure `as_bytes()` returns a valid
/// UTF-8 byte slice for the lifetime of `&self`.
unsafe trait RawBytes {
    fn as_bytes(&self) -> &[u8];
}

struct AsciiString(String);

// TODO: Implement the unsafe trait `RawBytes` for `AsciiString`.
// `as_bytes()` should return the underlying byte slice of the String.
// Mark the impl block with `unsafe` to acknowledge the safety contract.

// ============================================================
// Exercise 5: Dangling Pointer Detection (Conceptual)
// ============================================================
// Answer the following as comments:
//
// Q1: Why does this code NOT compile?
//   let r: *const i32;
//   {
//       let x = 5;
//       r = &x as *const i32;
//   }
//   unsafe { println!("{}", *r); }
//
// TODO: Write your answer here:
// A1:

// Q2: Why is it safe to split a &mut [T] into two non-overlapping
//     &mut [T] sub-slices using raw pointers, even though Rust
//     normally forbids two mutable references to the same data?
//
// TODO: Write your answer here:
// A2:

// ============================================================
// Main — run all exercises
// ============================================================

fn main() {
    println!("=== Exercise 1: Raw Pointer Basics ===");
    let a = 10_i32;
    let b = 32_i32;
    assert_eq!(raw_pointer_sum(&a, &b), 42);
    println!("raw_pointer_sum(10, 32) = {}", raw_pointer_sum(&a, &b));

    let mut x = 5_i32;
    let mut y = 99_i32;
    swap_via_raw(&mut x, &mut y);
    assert_eq!(x, 99);
    assert_eq!(y, 5);
    println!("swap_via_raw: x={x}, y={y}");

    println!("\n=== Exercise 2: Safe split_at_mut Wrapper ===");
    let mut data = vec![1, 2, 3, 4, 5];
    let (left, right) = split_at_mut_custom(&mut data, 3);
    assert_eq!(left, &[1, 2, 3]);
    assert_eq!(right, &[4, 5]);
    left[0] = 10;
    right[0] = 40;
    println!("After split and mutation: {:?}", {
        // re-borrow after split lifetimes end
        &data
    });

    println!("\n=== Exercise 3: Extern C abs ===");
    assert_eq!(safe_abs(-42), 42);
    assert_eq!(safe_abs(7), 7);
    assert_eq!(safe_abs(0), 0);
    println!("safe_abs(-42) = {}", safe_abs(-42));

    println!("\n=== Exercise 4: Unsafe Trait ===");
    let s = AsciiString("hello".to_string());
    let bytes = s.as_bytes();
    assert_eq!(bytes, b"hello");
    println!("AsciiString bytes: {:?}", bytes);

    println!("\n=== All exercises passed! ===");
}
