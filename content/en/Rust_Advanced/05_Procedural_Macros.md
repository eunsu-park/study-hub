# 21. Procedural Macros

**Previous**: [Declarative Macros](./04_Declarative_Macros.md) | **Next**: [Advanced Traits](./06_Advanced_Traits.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand the three types of procedural macros: derive, attribute, and function-like
2. Build a custom derive macro using the `syn` and `quote` crates
3. Create attribute macros that transform annotated items
4. Write function-like procedural macros for advanced code generation
5. Test and debug procedural macros effectively

---

Procedural macros are Rust's most powerful metaprogramming tool. Unlike declarative macros (`macro_rules!`) that match patterns, procedural macros are **Rust functions** that receive a token stream and produce a new token stream. They can do anything a Rust program can: parse complex syntax, query databases, read files, or generate arbitrarily complex code.

## Table of Contents
1. [Overview of Procedural Macros](#1-overview-of-procedural-macros)
2. [Project Setup](#2-project-setup)
3. [Token Streams](#3-token-streams)
4. [The syn Crate: Parsing](#4-the-syn-crate-parsing)
5. [The quote Crate: Code Generation](#5-the-quote-crate-code-generation)
6. [Derive Macros](#6-derive-macros)
7. [Attribute Macros](#7-attribute-macros)
8. [Function-like Macros](#8-function-like-macros)
9. [Error Handling in Proc Macros](#9-error-handling-in-proc-macros)
10. [Testing Procedural Macros](#10-testing-procedural-macros)
11. [Real-World Examples](#11-real-world-examples)
12. [Exercises](#12-exercises)

---

## 1. Overview of Procedural Macros

There are three kinds:

| Kind | Syntax | Input | Use Case |
|------|--------|-------|----------|
| **Derive** | `#[derive(MyMacro)]` | The struct/enum definition | Auto-implement traits |
| **Attribute** | `#[my_attribute]` | The annotated item | Transform or augment items |
| **Function-like** | `my_macro!(...)` | Arbitrary tokens | DSLs, complex generation |

### Key Difference from Declarative Macros

```
Declarative (macro_rules!):    Pattern → Template
Procedural:                    TokenStream → fn() → TokenStream
```

Procedural macros are compiled to a separate crate (a compiler plugin). They run as part of the compilation process.

---

## 2. Project Setup

Procedural macros **must** live in their own crate with `proc-macro = true`:

```bash
# Create a workspace
mkdir my-derive && cd my-derive
cargo init --name my-app

# Create the proc-macro crate
cargo new my-derive-macros --lib
```

The proc-macro crate's `Cargo.toml`:

```toml
# my-derive-macros/Cargo.toml
[package]
name = "my-derive-macros"
version = "0.1.0"
edition = "2021"

[lib]
proc-macro = true

[dependencies]
syn = { version = "2", features = ["full"] }
quote = "1"
proc-macro2 = "1"
```

The main crate depends on the macro crate:

```toml
# Cargo.toml
[package]
name = "my-app"
version = "0.1.0"
edition = "2021"

[dependencies]
my-derive-macros = { path = "./my-derive-macros" }
```

### Workspace Layout

```
my-derive/
├── Cargo.toml          ← workspace root or app crate
├── src/
│   └── main.rs
└── my-derive-macros/
    ├── Cargo.toml      ← proc-macro = true
    └── src/
        └── lib.rs      ← macro definitions
```

---

## 3. Token Streams

The core type is `proc_macro::TokenStream` — a sequence of tokens the compiler gives you:

```rust
// my-derive-macros/src/lib.rs
use proc_macro::TokenStream;

#[proc_macro]
pub fn hello_world(_input: TokenStream) -> TokenStream {
    // Return new code as a string parsed into tokens
    "fn hello() { println!(\"Hello from a proc macro!\"); }"
        .parse()
        .unwrap()
}
```

Usage:

```rust
// src/main.rs
use my_derive_macros::hello_world;

hello_world!();

fn main() {
    hello();  // Prints: Hello from a proc macro!
}
```

### TokenStream Contents

A token stream contains four types of tokens:

```rust
use proc_macro::TokenTree;

// Group: (...), [...], or {...}
// Ident: identifier like `foo`, `String`
// Punct: punctuation like `+`, `,`, `::`
// Literal: literal values like 42, "hello", 3.14
```

---

## 4. The syn Crate: Parsing

`syn` parses `TokenStream` into a structured AST. It's the standard tool for understanding Rust code:

```rust
use syn::{parse_macro_input, DeriveInput};

// Parse a derive macro's input into a structured type
#[proc_macro_derive(MyTrait)]
pub fn my_trait_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    // ast.ident — the name of the struct/enum
    // ast.generics — generic parameters
    // ast.data — struct fields or enum variants
    // ast.attrs — attributes on the item

    todo!()
}
```

### Key syn Types

```rust
use syn::{
    DeriveInput,      // Top-level struct/enum for derive macros
    Data,             // Enum: Struct, Enum, Union
    Fields,           // Named, Unnamed, or Unit
    Field,            // A single field with name, type, visibility
    Ident,            // An identifier
    Type,             // A type expression
    GenericParam,     // Generic type/lifetime/const parameters
    Attribute,        // An attribute like #[serde(rename = "foo")]
    Lit,              // A literal value
    Expr,             // An expression
};
```

### Inspecting Struct Fields

```rust
use syn::{Data, Fields};

fn get_field_names(data: &Data) -> Vec<&syn::Ident> {
    match data {
        Data::Struct(data_struct) => {
            match &data_struct.fields {
                Fields::Named(fields) => {
                    fields.named.iter()
                        .map(|f| f.ident.as_ref().unwrap())
                        .collect()
                }
                Fields::Unnamed(_) => vec![],  // Tuple struct
                Fields::Unit => vec![],         // Unit struct
            }
        }
        Data::Enum(_) => vec![],
        Data::Union(_) => vec![],
    }
}
```

---

## 5. The quote Crate: Code Generation

`quote` lets you write Rust code as a template with interpolation:

```rust
use quote::quote;
use syn::Ident;

fn generate_greeting(name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        impl #name {
            pub fn greet(&self) -> String {
                format!("Hello from {}!", stringify!(#name))
            }
        }
    }
}
```

### Interpolation with `#variable`

```rust
use quote::quote;
use proc_macro2::Span;
use syn::Ident;

let struct_name = Ident::new("MyStruct", Span::call_site());
let field_count = 3usize;

let tokens = quote! {
    impl #struct_name {
        pub fn field_count(&self) -> usize {
            #field_count
        }
    }
};
```

### Repetition with `#( ... )*`

```rust
use quote::quote;
use syn::Ident;

let field_names: Vec<Ident> = vec![
    Ident::new("name", proc_macro2::Span::call_site()),
    Ident::new("age", proc_macro2::Span::call_site()),
];

let tokens = quote! {
    impl MyStruct {
        pub fn field_names() -> Vec<&'static str> {
            vec![ #( stringify!(#field_names) ),* ]
        }
    }
};
// Generates: vec!["name", "age"]
```

---

## 6. Derive Macros

Derive macros are the most common type. They implement traits automatically:

### Example: Custom Debug-like Trait

```rust
// my-derive-macros/src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

#[proc_macro_derive(Describe)]
pub fn describe_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    // Extract field names for structs with named fields
    let field_descriptions = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let field_strs: Vec<_> = fields.named.iter().map(|f| {
                    let fname = f.ident.as_ref().unwrap();
                    let ftype = &f.ty;
                    quote! {
                        format!("  {}: {} = {:?}", stringify!(#fname), stringify!(#ftype), self.#fname)
                    }
                }).collect();

                quote! {
                    let fields = vec![ #( #field_strs ),* ];
                    fields.join("\n")
                }
            }
            Fields::Unnamed(fields) => {
                let indices: Vec<_> = (0..fields.unnamed.len())
                    .map(syn::Index::from)
                    .collect();
                quote! {
                    let fields = vec![ #( format!("  .{}: {:?}", #indices, self.#indices) ),* ];
                    fields.join("\n")
                }
            }
            Fields::Unit => quote! { String::new() },
        },
        _ => quote! { String::new() },
    };

    let expanded = quote! {
        impl #name {
            pub fn describe(&self) -> String {
                let fields = { #field_descriptions };
                if fields.is_empty() {
                    format!("{} (no fields)", stringify!(#name))
                } else {
                    format!("{} {{\n{}\n}}", stringify!(#name), fields)
                }
            }
        }
    };

    TokenStream::from(expanded)
}
```

Usage:

```rust
use my_derive_macros::Describe;

#[derive(Describe, Debug)]
struct User {
    name: String,
    age: u32,
    active: bool,
}

fn main() {
    let user = User {
        name: "Alice".into(),
        age: 30,
        active: true,
    };
    println!("{}", user.describe());
    // User {
    //   name: String = "Alice"
    //   age: u32 = 30
    //   active: bool = true
    // }
}
```

### Derive with Helper Attributes

```rust
// Define a derive macro with helper attributes
#[proc_macro_derive(Validate, attributes(validate))]
pub fn validate_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let validations = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let checks: Vec<_> = fields.named.iter().filter_map(|f| {
                    let fname = f.ident.as_ref().unwrap();
                    // Look for #[validate(non_empty)] attribute
                    let has_non_empty = f.attrs.iter().any(|attr| {
                        attr.path().is_ident("validate") &&
                        attr.parse_args::<syn::Ident>()
                            .map(|id| id == "non_empty")
                            .unwrap_or(false)
                    });

                    if has_non_empty {
                        Some(quote! {
                            if self.#fname.is_empty() {
                                errors.push(format!("{} must not be empty", stringify!(#fname)));
                            }
                        })
                    } else {
                        None
                    }
                }).collect();

                quote! { #( #checks )* }
            }
            _ => quote! {},
        },
        _ => quote! {},
    };

    let expanded = quote! {
        impl #name {
            pub fn validate(&self) -> Result<(), Vec<String>> {
                let mut errors = Vec::new();
                #validations
                if errors.is_empty() {
                    Ok(())
                } else {
                    Err(errors)
                }
            }
        }
    };

    TokenStream::from(expanded)
}
```

Usage:

```rust
#[derive(Validate)]
struct Registration {
    #[validate(non_empty)]
    username: String,
    #[validate(non_empty)]
    email: String,
    bio: String,  // No validation
}
```

---

## 7. Attribute Macros

Attribute macros transform the item they're attached to:

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Times the execution of a function and prints the duration
#[proc_macro_attribute]
pub fn timed(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let name = &func.sig.ident;
    let block = &func.block;
    let sig = &func.sig;
    let vis = &func.vis;
    let attrs = &func.attrs;

    let expanded = quote! {
        #( #attrs )*
        #vis #sig {
            let __start = std::time::Instant::now();
            let __result = (|| #block)();
            let __elapsed = __start.elapsed();
            println!("[{}] executed in {:?}", stringify!(#name), __elapsed);
            __result
        }
    };

    TokenStream::from(expanded)
}
```

Usage:

```rust
#[timed]
fn expensive_computation() -> u64 {
    (0..1_000_000).sum()
}

fn main() {
    let result = expensive_computation();
    // Prints: [expensive_computation] executed in 1.234ms
    println!("Result: {result}");
}
```

### Attribute with Arguments

```rust
use syn::{parse_macro_input, LitStr, ItemFn};

#[proc_macro_attribute]
pub fn route(attr: TokenStream, item: TokenStream) -> TokenStream {
    let path = parse_macro_input!(attr as LitStr);
    let func = parse_macro_input!(item as ItemFn);
    let name = &func.sig.ident;

    let expanded = quote! {
        #func

        // Register this handler at compile time (conceptual)
        inventory::submit! {
            Route {
                path: #path,
                handler: #name,
            }
        }
    };

    TokenStream::from(expanded)
}

// Usage:
// #[route("/api/users")]
// fn get_users() -> Response { ... }
```

---

## 8. Function-like Macros

Function-like proc macros look like regular macro invocations but have full procedural power:

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitStr};

/// SQL query validation at compile time (conceptual)
#[proc_macro]
pub fn sql(input: TokenStream) -> TokenStream {
    let query = parse_macro_input!(input as LitStr);
    let query_str = query.value();

    // Validate SQL at compile time
    if !query_str.to_uppercase().starts_with("SELECT") &&
       !query_str.to_uppercase().starts_with("INSERT") &&
       !query_str.to_uppercase().starts_with("UPDATE") &&
       !query_str.to_uppercase().starts_with("DELETE") {
        return syn::Error::new(query.span(), "Invalid SQL: must start with SELECT, INSERT, UPDATE, or DELETE")
            .to_compile_error()
            .into();
    }

    let expanded = quote! {
        {
            // At runtime, just return the validated query string
            #query_str.to_string()
        }
    };

    TokenStream::from(expanded)
}

// Usage:
// let query = sql!("SELECT * FROM users WHERE id = ?");
// let bad = sql!("DROP TABLE users");  // Compile error!
```

### Custom Parsing

```rust
use syn::parse::{Parse, ParseStream};
use syn::{Token, Ident, LitStr, Result};

// Define custom syntax: key => value, key => value, ...
struct KeyValue {
    key: Ident,
    value: LitStr,
}

impl Parse for KeyValue {
    fn parse(input: ParseStream) -> Result<Self> {
        let key: Ident = input.parse()?;
        input.parse::<Token![=>]>()?;
        let value: LitStr = input.parse()?;
        Ok(KeyValue { key, value })
    }
}

struct KeyValueList {
    items: Vec<KeyValue>,
}

impl Parse for KeyValueList {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut items = Vec::new();
        while !input.is_empty() {
            items.push(input.parse()?);
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(KeyValueList { items })
    }
}

#[proc_macro]
pub fn env_config(input: TokenStream) -> TokenStream {
    let list = parse_macro_input!(input as KeyValueList);

    let fields: Vec<_> = list.items.iter().map(|kv| {
        let key = &kv.key;
        let env_var = &kv.value;
        quote! {
            pub #key: String
        }
    }).collect();

    let inits: Vec<_> = list.items.iter().map(|kv| {
        let key = &kv.key;
        let env_var = &kv.value;
        quote! {
            #key: std::env::var(#env_var)
                .unwrap_or_else(|_| String::new())
        }
    }).collect();

    let expanded = quote! {
        pub struct EnvConfig {
            #( #fields, )*
        }

        impl EnvConfig {
            pub fn load() -> Self {
                Self {
                    #( #inits, )*
                }
            }
        }
    };

    TokenStream::from(expanded)
}

// Usage:
// env_config!(
//     database_url => "DATABASE_URL",
//     api_key => "API_KEY",
//     port => "PORT"
// );
```

---

## 9. Error Handling in Proc Macros

Good error messages are critical for macro usability:

```rust
use syn::Error;

#[proc_macro_derive(MyMacro)]
pub fn my_macro(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    // Reject enums
    if let Data::Enum(_) = &ast.data {
        return Error::new_spanned(&ast.ident, "MyMacro does not support enums")
            .to_compile_error()
            .into();
    }

    // Reject generic types
    if !ast.generics.params.is_empty() {
        return Error::new_spanned(
            &ast.generics,
            "MyMacro does not support generic types"
        )
            .to_compile_error()
            .into();
    }

    // ... generate code
    TokenStream::new()
}
```

### Combining Multiple Errors

```rust
fn validate_fields(fields: &Fields) -> Result<(), TokenStream> {
    let mut errors = Vec::new();

    if let Fields::Named(named) = fields {
        for field in &named.named {
            let fname = field.ident.as_ref().unwrap();
            if fname.to_string().starts_with('_') {
                errors.push(
                    Error::new_spanned(fname, "Fields must not start with underscore")
                );
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        // Combine all errors into one
        let mut combined = errors[0].clone();
        for err in &errors[1..] {
            combined.combine(err.clone());
        }
        Err(combined.to_compile_error().into())
    }
}
```

---

## 10. Testing Procedural Macros

### Using trybuild for Compile-Test

```toml
[dev-dependencies]
trybuild = "1"
```

```rust
#[test]
fn tests() {
    let t = trybuild::TestCases::new();
    t.pass("tests/01-basic.rs");        // Should compile
    t.compile_fail("tests/02-error.rs"); // Should fail with expected error
}
```

Test files:

```rust
// tests/01-basic.rs
use my_derive_macros::Describe;

#[derive(Describe, Debug)]
struct Point {
    x: f64,
    y: f64,
}

fn main() {
    let p = Point { x: 1.0, y: 2.0 };
    let desc = p.describe();
    assert!(desc.contains("x"));
    assert!(desc.contains("y"));
}
```

```rust
// tests/02-error.rs
use my_derive_macros::Describe;

#[derive(Describe)]
enum Color { Red, Green, Blue }  // Should fail: enums not supported

fn main() {}
```

### Unit Testing with quote and syn

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn test_field_extraction() {
        let input = quote! {
            struct Foo {
                name: String,
                age: u32,
            }
        };

        let ast: DeriveInput = syn::parse2(input).unwrap();
        let fields = get_field_names(&ast.data);
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].to_string(), "name");
        assert_eq!(fields[1].to_string(), "age");
    }
}
```

---

## 11. Real-World Examples

### How serde Works (Simplified)

```rust
// When you write:
#[derive(Serialize)]
struct User {
    name: String,
    age: u32,
}

// serde's proc macro generates approximately:
impl Serialize for User {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("User", 2)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("age", &self.age)?;
        state.end()
    }
}
```

### How thiserror Works (Simplified)

```rust
// When you write:
#[derive(thiserror::Error, Debug)]
enum AppError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },
}

// thiserror generates Display impl, From impls, and Error impl
```

### How clap Derive Works (Simplified)

```rust
// When you write:
#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value = "8080")]
    port: u16,
}

// clap generates argument parsing code that reads from std::env::args()
```

---

## 12. Exercises

1. **Derive `IntoMap`**: Write a derive macro that implements `fn into_map(&self) -> HashMap<String, String>` for any struct with named fields, converting each field to its `Debug` representation.

2. **Attribute `#[log_calls]`**: Write an attribute macro that wraps a function to print its name and arguments on entry, and its return value on exit.

3. **Function-like `html!`**: Write a proc macro that validates a simple HTML-like syntax at compile time: `html!(<div class="foo"><p>Hello</p></div>)` should produce a `String`.

4. **Derive with generics**: Extend the `Describe` macro to handle structs with generic type parameters (e.g., `struct Wrapper<T> { inner: T }`).

5. **Error diagnostics**: Improve one of your macros to produce helpful error messages with span information pointing to the exact problematic token.

---

## References

- [The Rust Reference: Procedural Macros](https://doc.rust-lang.org/reference/procedural-macros.html)
- [syn documentation](https://docs.rs/syn/latest/syn/)
- [quote documentation](https://docs.rs/quote/latest/quote/)
- [proc-macro2 documentation](https://docs.rs/proc-macro2/latest/proc_macro2/)
- [trybuild documentation](https://docs.rs/trybuild/latest/trybuild/)
- [dtolnay's proc macro workshop](https://github.com/dtolnay/proc-macro-workshop)

---

**Previous**: [Declarative Macros](./04_Declarative_Macros.md) | **Next**: [Advanced Traits](./06_Advanced_Traits.md)
