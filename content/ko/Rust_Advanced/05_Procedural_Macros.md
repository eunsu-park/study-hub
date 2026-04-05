# 21. 절차적 매크로

**이전**: [선언적 매크로](./04_Declarative_Macros.md) | **다음**: [고급 트레이트](./06_Advanced_Traits.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 절차적 매크로의 세 가지 유형(디라이브, 속성, 함수형) 이해하기
2. `syn`과 `quote` 크레이트를 사용하여 커스텀 디라이브 매크로 작성하기
3. 어노테이션된 아이템을 변환하는 속성 매크로 만들기
4. 고급 코드 생성을 위한 함수형 절차적 매크로 작성하기
5. 절차적 매크로를 효과적으로 테스트하고 디버깅하기

---

절차적 매크로는 Rust의 가장 강력한 메타프로그래밍 도구입니다. 패턴을 매칭하는 선언적 매크로(`macro_rules!`)와 달리, 절차적 매크로는 토큰 스트림을 받아 새로운 토큰 스트림을 생성하는 **Rust 함수**입니다. Rust 프로그램이 할 수 있는 모든 것을 할 수 있습니다: 복잡한 구문 파싱, 데이터베이스 쿼리, 파일 읽기, 임의로 복잡한 코드 생성 등.

## 목차
1. [절차적 매크로 개요](#1-절차적-매크로-개요)
2. [프로젝트 설정](#2-프로젝트-설정)
3. [토큰 스트림](#3-토큰-스트림)
4. [syn 크레이트: 파싱](#4-syn-크레이트-파싱)
5. [quote 크레이트: 코드 생성](#5-quote-크레이트-코드-생성)
6. [디라이브 매크로](#6-디라이브-매크로)
7. [속성 매크로](#7-속성-매크로)
8. [함수형 매크로](#8-함수형-매크로)
9. [절차적 매크로의 에러 처리](#9-절차적-매크로의-에러-처리)
10. [절차적 매크로 테스팅](#10-절차적-매크로-테스팅)
11. [실제 예제](#11-실제-예제)
12. [연습문제](#12-연습문제)

---

## 1. 절차적 매크로 개요

세 가지 종류가 있습니다:

| 종류 | 구문 | 입력 | 용도 |
|------|------|------|------|
| **디라이브** | `#[derive(MyMacro)]` | 구조체/열거형 정의 | 트레이트 자동 구현 |
| **속성** | `#[my_attribute]` | 어노테이션된 아이템 | 아이템 변환 또는 보강 |
| **함수형** | `my_macro!(...)` | 임의의 토큰 | DSL, 복잡한 생성 |

### 선언적 매크로와의 핵심 차이점

```
선언적 (macro_rules!):    패턴 → 템플릿
절차적:                   TokenStream → fn() → TokenStream
```

절차적 매크로는 `proc-macro = true`인 별도의 크레이트로 컴파일됩니다(컴파일러 플러그인). 컴파일 과정의 일부로 실행됩니다.

---

## 2. 프로젝트 설정

절차적 매크로는 **반드시** `proc-macro = true`인 자체 크레이트에 있어야 합니다:

```bash
# 워크스페이스 생성
mkdir my-derive && cd my-derive
cargo init --name my-app

# proc-macro 크레이트 생성
cargo new my-derive-macros --lib
```

proc-macro 크레이트의 `Cargo.toml`:

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

---

## 3. 토큰 스트림

핵심 타입은 `proc_macro::TokenStream`입니다 — 컴파일러가 제공하는 토큰의 시퀀스:

```rust
use proc_macro::TokenStream;

#[proc_macro]
pub fn hello_world(_input: TokenStream) -> TokenStream {
    "fn hello() { println!(\"Hello from a proc macro!\"); }"
        .parse()
        .unwrap()
}
```

사용법:

```rust
use my_derive_macros::hello_world;

hello_world!();

fn main() {
    hello();  // 출력: Hello from a proc macro!
}
```

---

## 4. syn 크레이트: 파싱

`syn`은 `TokenStream`을 구조화된 AST로 파싱합니다. Rust 코드를 이해하기 위한 표준 도구입니다:

```rust
use syn::{parse_macro_input, DeriveInput};

// 디라이브 매크로의 입력을 구조화된 타입으로 파싱
#[proc_macro_derive(MyTrait)]
pub fn my_trait_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    // ast.ident — 구조체/열거형의 이름
    // ast.generics — 제네릭 파라미터
    // ast.data — 구조체 필드 또는 열거형 배리언트
    // ast.attrs — 아이템의 속성

    todo!()
}
```

### 주요 syn 타입

```rust
use syn::{
    DeriveInput,      // 디라이브 매크로를 위한 최상위 구조체/열거형
    Data,             // 열거형: Struct, Enum, Union
    Fields,           // Named, Unnamed, 또는 Unit
    Field,            // 이름, 타입, 가시성이 있는 단일 필드
    Ident,            // 식별자
    Type,             // 타입 표현식
    GenericParam,     // 제네릭 타입/라이프타임/const 파라미터
    Attribute,        // #[serde(rename = "foo")] 같은 속성
    Lit,              // 리터럴 값
    Expr,             // 표현식
};
```

### 구조체 필드 검사

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
                Fields::Unnamed(_) => vec![],  // 튜플 구조체
                Fields::Unit => vec![],         // 유닛 구조체
            }
        }
        Data::Enum(_) => vec![],
        Data::Union(_) => vec![],
    }
}
```

---

## 5. quote 크레이트: 코드 생성

`quote`는 보간(interpolation)이 있는 템플릿으로 Rust 코드를 작성할 수 있게 합니다:

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

### `#variable`로 보간

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

### `#( ... )*`로 반복

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
// 생성 결과: vec!["name", "age"]
```

---

## 6. 디라이브 매크로

디라이브 매크로는 가장 일반적인 유형입니다. 트레이트를 자동으로 구현합니다:

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

#[proc_macro_derive(Describe)]
pub fn describe_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

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
            _ => quote! { String::new() },
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

사용법:

```rust
use my_derive_macros::Describe;

#[derive(Describe, Debug)]
struct User {
    name: String,
    age: u32,
    active: bool,
}

fn main() {
    let user = User { name: "Alice".into(), age: 30, active: true };
    println!("{}", user.describe());
}
```

---

## 7. 속성 매크로

속성 매크로는 연결된 아이템을 변환합니다:

```rust
#[proc_macro_attribute]
pub fn timed(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let name = &func.sig.ident;
    let block = &func.block;
    let sig = &func.sig;
    let vis = &func.vis;

    let expanded = quote! {
        #vis #sig {
            let __start = std::time::Instant::now();
            let __result = (|| #block)();
            println!("[{}] executed in {:?}", stringify!(#name), __start.elapsed());
            __result
        }
    };

    TokenStream::from(expanded)
}
```

---

## 8. 함수형 매크로

함수형 절차적 매크로는 일반 매크로 호출처럼 보이지만 완전한 절차적 능력을 가집니다:

```rust
#[proc_macro]
pub fn sql(input: TokenStream) -> TokenStream {
    let query = parse_macro_input!(input as LitStr);
    let query_str = query.value();

    if !query_str.to_uppercase().starts_with("SELECT") &&
       !query_str.to_uppercase().starts_with("INSERT") {
        return syn::Error::new(query.span(), "Invalid SQL statement")
            .to_compile_error()
            .into();
    }

    let expanded = quote! { #query_str.to_string() };
    TokenStream::from(expanded)
}
```

---

## 9. 절차적 매크로의 에러 처리

좋은 에러 메시지는 매크로 사용성에 매우 중요합니다:

```rust
use syn::Error;

#[proc_macro_derive(MyMacro)]
pub fn my_macro(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    if let Data::Enum(_) = &ast.data {
        return Error::new_spanned(&ast.ident, "MyMacro는 열거형을 지원하지 않습니다")
            .to_compile_error()
            .into();
    }

    TokenStream::new()
}
```

---

## 10. 절차적 매크로 테스팅

### trybuild로 컴파일 테스트

```toml
[dev-dependencies]
trybuild = "1"
```

```rust
#[test]
fn tests() {
    let t = trybuild::TestCases::new();
    t.pass("tests/01-basic.rs");        // 컴파일 성공해야 함
    t.compile_fail("tests/02-error.rs"); // 예상된 에러로 실패해야 함
}
```

테스트 파일:

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
enum Color { Red, Green, Blue }  // 실패해야 함: 열거형 미지원

fn main() {}
```

### quote와 syn으로 유닛 테스트

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

## 11. 실제 예제

### serde 작동 방식 (간략화)

```rust
// 이렇게 작성하면:
#[derive(Serialize)]
struct User {
    name: String,
    age: u32,
}

// serde의 절차적 매크로가 대략 다음을 생성합니다:
impl Serialize for User {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("User", 2)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("age", &self.age)?;
        state.end()
    }
}
```

### thiserror 작동 방식 (간략화)

```rust
// 이렇게 작성하면:
#[derive(thiserror::Error, Debug)]
enum AppError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },
}

// thiserror가 Display 구현, From 구현, Error 구현을 생성합니다
```

### clap 디라이브 작동 방식 (간략화)

```rust
// 이렇게 작성하면:
#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value = "8080")]
    port: u16,
}

// clap이 std::env::args()에서 읽는 인수 파싱 코드를 생성합니다
```

---

## 12. 연습문제

1. **`IntoMap` 디라이브**: 명명된 필드가 있는 구조체에 대해 `fn into_map(&self) -> HashMap<String, String>`을 구현하는 디라이브 매크로를 작성하세요.

2. **`#[log_calls]` 속성**: 함수를 래핑하여 진입 시 이름과 인수를, 종료 시 반환 값을 출력하는 속성 매크로를 작성하세요.

3. **`html!` 함수형**: 컴파일 타임에 간단한 HTML 스타일 구문을 검증하는 절차적 매크로를 작성하세요.

4. **제네릭이 있는 디라이브**: `Describe` 매크로를 제네릭 타입 파라미터가 있는 구조체를 처리하도록 확장하세요.

5. **에러 진단**: 매크로 중 하나를 개선하여 정확한 문제 토큰을 가리키는 스팬 정보와 함께 유용한 에러 메시지를 생성하세요.

---

## 참고 자료

- [The Rust Reference: Procedural Macros](https://doc.rust-lang.org/reference/procedural-macros.html)
- [syn documentation](https://docs.rs/syn/latest/syn/)
- [quote documentation](https://docs.rs/quote/latest/quote/)
- [proc-macro2 documentation](https://docs.rs/proc-macro2/latest/proc_macro2/)
- [trybuild documentation](https://docs.rs/trybuild/latest/trybuild/)
- [dtolnay's proc macro workshop](https://github.com/dtolnay/proc-macro-workshop)

---

**이전**: [선언적 매크로](./04_Declarative_Macros.md) | **다음**: [고급 트레이트](./06_Advanced_Traits.md)
