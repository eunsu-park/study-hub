// Exercise: Procedural Macros (Conceptual)
// Procedural macros require a separate crate. This file contains
// conceptual exercises and tests you would write.
//
// To practice, create a workspace as described in Lesson 21.

// Exercise 1: Describe the three types of procedural macros
// and when to use each one.
//
// - Derive macros: #[derive(MyTrait)] — auto-implement traits
// - Attribute macros: #[my_attr] — transform items
// - Function-like macros: my_macro!(...) — DSLs and complex generation

// Exercise 2: Sketch a derive macro that generates IntoMap
// Given: #[derive(IntoMap)]
//        struct User { name: String, age: u32 }
// Generate: impl User { fn into_map(&self) -> HashMap<String, String> { ... } }

// Exercise 3: What is the role of each crate?
// - proc_macro: Compiler interface for proc macros
// - syn: Parse TokenStream into AST
// - quote: Generate TokenStream from Rust-like templates
// - proc-macro2: Wrapper for testability

// Exercise 4: Write pseudo-code for a #[log_calls] attribute macro
// that wraps a function to print entry/exit messages.

// Exercise 5: How would you test a procedural macro?
// - Use trybuild for compile-pass and compile-fail tests
// - Use syn::parse2 for unit testing parsing logic
// - Use quote! to create test inputs

fn main() {
    println!("Procedural macro exercises are conceptual.");
    println!("Create a workspace with a proc-macro crate to practice.");
    println!("See Lesson 21 for step-by-step instructions.");
}
