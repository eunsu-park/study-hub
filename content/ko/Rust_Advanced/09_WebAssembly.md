# 25. WebAssembly

**이전**: [FFI와 상호운용](./08_FFI_and_Interop.md) | **다음**: [임베디드 Rust](./10_Embedded_Rust.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `wasm-pack`과 `wasm-bindgen`을 사용하여 Rust를 WebAssembly로 컴파일하기
2. `wasm-bindgen`과 `web-sys`를 통해 Rust에서 JavaScript API와 상호작용하기
3. 브라우저와 WASI(서버 사이드) 환경 모두를 대상으로 애플리케이션 빌드하기
4. Yew 프레임워크로 완전한 웹 프론트엔드 애플리케이션 구축하기
5. Wasm 바이너리 크기 최적화 및 Wasm 애플리케이션 디버깅하기

---

WebAssembly(Wasm)는 웹 브라우저에서 JavaScript와 함께 실행되고, WASI를 통해 점점 더 서버 사이드 환경에서도 실행되는 바이너리 명령어 형식입니다. Rust는 작은 런타임, 가비지 컬렉터 없음, 우수한 도구 지원으로 Wasm을 타겟으로 하는 최고의 언어 중 하나입니다.

## 목차
1. [WebAssembly 개요](#1-webassembly-개요)
2. [설정 및 도구](#2-설정-및-도구)
3. [wasm-bindgen 기초](#3-wasm-bindgen-기초)
4. [JavaScript와 상호작용](#4-javascript와-상호작용)
5. [web-sys와 js-sys](#5-web-sys와-js-sys)
6. [DOM 조작](#6-dom-조작)
7. [WASI: 서버 사이드 Wasm](#7-wasi-서버-사이드-wasm)
8. [Yew 프레임워크](#8-yew-프레임워크)
9. [바이너리 크기 최적화](#9-바이너리-크기-최적화)
10. [Wasm 디버깅](#10-wasm-디버깅)
11. [실용적 패턴](#11-실용적-패턴)
12. [연습문제](#12-연습문제)

---

## 1. WebAssembly 개요

```
┌──────────────────────────────────────┐
│            웹 브라우저               │
│  ┌─────────────┐  ┌──────────────┐  │
│  │ JavaScript  │◄►│  Wasm 모듈   │  │
│  │   엔진      │  │  (Rust에서)  │  │
│  └─────────────┘  └──────────────┘  │
│         │                  │         │
│         └──────┬───────────┘         │
│                ▼                     │
│           Web API                    │
│  (DOM, fetch, Canvas, WebGL 등)      │
└──────────────────────────────────────┘
```

핵심 특성:
- **컴팩트 바이너리 형식** — 동등한 코드의 JavaScript보다 작음
- **네이티브에 가까운 속도** — 사전 컴파일(AOT), 예측 가능한 성능
- **샌드박스** — JavaScript와 동일한 보안 샌드박스에서 실행
- **언어 독립적** — Rust, C, C++, Go 등에서 타겟 가능

### Wasm과 JavaScript 트레이드오프

| 측면 | JavaScript | Wasm (Rust) |
|------|-----------|-------------|
| 시작 | 빠름 (JIT) | 빠름 (AOT) |
| 최대 성능 | 좋음 (JIT 최적화) | 탁월 (네이티브에 가까움) |
| DOM 접근 | 직접 | JS 브릿지를 통해 |
| 번들 크기 | 작음 (텍스트) | 작음 (바이너리) |
| GC 중단 | 있음 | 없음 |
| 최적 용도 | UI, DOM, 글루 코드 | 연산, 코덱, 게임 |

---

## 2. 설정 및 도구

```bash
# Wasm 타겟 설치
rustup target add wasm32-unknown-unknown

# wasm-pack 설치 (Wasm 패키지 빌드, 테스트, 게시)
cargo install wasm-pack

# WASI 타겟용
rustup target add wasm32-wasip1

# 선택사항: WASI용 wasmtime 런타임
cargo install wasmtime-cli
```

### 프로젝트 설정

```bash
# 새 라이브러리 프로젝트 생성
cargo new --lib my-wasm-lib
cd my-wasm-lib
```

```toml
# Cargo.toml
[package]
name = "my-wasm-lib"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"

[dev-dependencies]
wasm-bindgen-test = "0.3"

[profile.release]
opt-level = "s"     # 크기 최적화
lto = true          # 링크 타임 최적화
```

---

## 3. wasm-bindgen 기초

`wasm-bindgen`은 Rust와 JavaScript를 연결하고 타입 변환을 자동으로 처리합니다:

```rust
use wasm_bindgen::prelude::*;

// JavaScript에 함수 내보내기
#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("안녕하세요, {name}!")
}

// 구조체 내보내기
#[wasm_bindgen]
pub struct Calculator {
    value: f64,
}

#[wasm_bindgen]
impl Calculator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Calculator {
        Calculator { value: 0.0 }
    }

    pub fn add(&mut self, n: f64) {
        self.value += n;
    }

    pub fn subtract(&mut self, n: f64) {
        self.value -= n;
    }

    pub fn multiply(&mut self, n: f64) {
        self.value *= n;
    }

    pub fn result(&self) -> f64 {
        self.value
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
    }
}
```

빌드 및 사용:

```bash
wasm-pack build --target web
```

```html
<!DOCTYPE html>
<html>
<body>
<script type="module">
  import init, { greet, Calculator } from './pkg/my_wasm_lib.js';

  async function main() {
    await init();

    console.log(greet("세계"));  // "안녕하세요, 세계!"

    const calc = new Calculator();
    calc.add(10);
    calc.multiply(3);
    calc.subtract(5);
    console.log(`결과: ${calc.result()}`);  // 25
    calc.free();  // Wasm 메모리 해제
  }

  main();
</script>
</body>
</html>
```

### 빌드 타겟

```bash
# 번들러 사용 (webpack, vite 등)
wasm-pack build --target bundler

# 브라우저에서 직접 사용 (ES 모듈)
wasm-pack build --target web

# Node.js용
wasm-pack build --target nodejs
```

---

## 4. JavaScript와 상호작용

### JavaScript 함수 가져오기

```rust
use wasm_bindgen::prelude::*;

// JavaScript 함수 가져오기
#[wasm_bindgen]
extern "C" {
    // console.log
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    // console.warn
    #[wasm_bindgen(js_namespace = console, js_name = warn)]
    fn console_warn(s: &str);

    // window.alert
    fn alert(s: &str);

    // 커스텀 JS 함수 가져오기
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;

    // JS 클래스 가져오기
    type Date;
    #[wasm_bindgen(constructor)]
    fn new() -> Date;
    #[wasm_bindgen(method, js_name = toISOString)]
    fn to_iso_string(this: &Date) -> String;
}

#[wasm_bindgen]
pub fn demo() {
    log("Rust에서 안녕하세요!");
    console_warn("이것은 경고입니다");

    let date = Date::new();
    log(&format!("현재 시각: {}", date.to_iso_string()));
    log(&format!("랜덤 숫자: {}", random()));
}
```

### 복잡한 타입 전달

```rust
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct UserData {
    pub name: String,
    pub age: u32,
    pub scores: Vec<f64>,
}

// 복잡한 타입 변환에 serde 사용
#[wasm_bindgen]
pub fn process_user(val: JsValue) -> Result<JsValue, JsValue> {
    let user: UserData = serde_wasm_bindgen::from_value(val)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let avg_score: f64 = user.scores.iter().sum::<f64>() / user.scores.len() as f64;

    let result = serde_json::json!({
        "name": user.name,
        "average_score": avg_score,
        "grade": if avg_score >= 90.0 { "A" } else if avg_score >= 80.0 { "B" } else { "C" }
    });

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
```

---

## 5. web-sys와 js-sys

`web-sys`는 Web API에 대한 바인딩을 제공합니다. `js-sys`는 JavaScript 내장 객체에 대한 바인딩을 제공합니다:

```toml
[dependencies]
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = [
    "Document", "Element", "HtmlElement", "Window",
    "console", "HtmlCanvasElement", "CanvasRenderingContext2d",
    "Request", "RequestInit", "Response", "Headers",
] }
js-sys = "0.3"
```

```rust
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, Window};

fn window() -> Window {
    web_sys::window().expect("글로벌 `window` 없음")
}

fn document() -> Document {
    window().document().expect("`document` 없음")
}

#[wasm_bindgen]
pub fn create_paragraph(text: &str) -> Result<(), JsValue> {
    let document = document();
    let body = document.body().expect("body 없음");

    let p = document.create_element("p")?;
    p.set_text_content(Some(text));
    p.set_attribute("class", "rust-paragraph")?;
    body.append_child(&p)?;

    Ok(())
}

// js-sys를 사용하여 JavaScript 내장 타입 사용
use js_sys::{Array, Date, Map, Promise};

#[wasm_bindgen]
pub fn js_types_demo() {
    // JS 배열 생성
    let arr = Array::new();
    arr.push(&JsValue::from(1));
    arr.push(&JsValue::from(2));
    arr.push(&JsValue::from(3));

    web_sys::console::log_1(&format!("배열 길이: {}", arr.length()).into());

    // JS Date
    let now = Date::new_0();
    web_sys::console::log_1(&format!("시각: {}", now.to_iso_string()).into());

    // JS Map
    let map = Map::new();
    map.set(&"key".into(), &"value".into());
}
```

---

## 6. DOM 조작

### Canvas 그리기

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};
use std::f64::consts::PI;

#[wasm_bindgen]
pub fn draw_chart(canvas_id: &str, data: &[f64]) -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .get_element_by_id(canvas_id)
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()?;

    let ctx = canvas
        .get_context("2d")?
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()?;

    let width = canvas.width() as f64;
    let height = canvas.height() as f64;

    // 캔버스 지우기
    ctx.clear_rect(0.0, 0.0, width, height);

    // 막대 차트 그리기
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bar_width = width / data.len() as f64 * 0.8;
    let gap = width / data.len() as f64 * 0.2;

    let colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"];

    for (i, &value) in data.iter().enumerate() {
        let bar_height = (value / max_val) * (height * 0.8);
        let x = i as f64 * (bar_width + gap) + gap;
        let y = height - bar_height;

        ctx.set_fill_style_str(colors[i % colors.len()]);
        ctx.fill_rect(x, y, bar_width, bar_height);

        // 레이블
        ctx.set_fill_style_str("#333");
        ctx.set_font("14px sans-serif");
        ctx.set_text_align("center");
        ctx.fill_text(
            &format!("{:.0}", value),
            x + bar_width / 2.0,
            y - 5.0,
        )?;
    }

    Ok(())
}
```

### 이벤트 처리

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen]
pub fn setup_click_handler(button_id: &str) -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let button = document.get_element_by_id(button_id).unwrap();

    let closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let x = event.client_x();
        let y = event.client_y();
        web_sys::console::log_1(&format!("({x}, {y})에서 클릭").into());
    }) as Box<dyn FnMut(_)>);

    button.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;

    // 중요: 클로저가 드롭되지 않도록 방지
    closure.forget();

    Ok(())
}
```

---

## 7. WASI: 서버 사이드 Wasm

WASI(WebAssembly System Interface)는 Wasm이 시스템 리소스에 제어된 접근으로 브라우저 외부에서 실행되게 합니다:

```rust
// 간단한 WASI 프로그램 — 파일 I/O와 환경
use std::env;
use std::fs;

fn main() {
    // 환경 변수
    for (key, value) in env::vars() {
        println!("{key}={value}");
    }

    // 커맨드라인 인수
    let args: Vec<String> = env::args().collect();
    println!("인수: {args:?}");

    // 파일 I/O (샌드박스)
    let content = "WASI Rust에서 안녕하세요!\n";
    fs::write("output.txt", content).expect("쓰기 실패");

    let read_back = fs::read_to_string("output.txt").expect("읽기 실패");
    println!("읽음: {read_back}");

    // 현재 시각
    let now = std::time::SystemTime::now();
    println!("시각: {now:?}");
}
```

빌드 및 실행:

```bash
# WASI용 빌드
cargo build --target wasm32-wasip1 --release

# wasmtime으로 실행
wasmtime target/wasm32-wasip1/release/my-wasi-app.wasm

# 디렉토리 접근 포함 (샌드박스)
wasmtime --dir=./data target/wasm32-wasip1/release/my-wasi-app.wasm

# 환경 변수 포함
wasmtime --env FOO=bar target/wasm32-wasip1/release/my-wasi-app.wasm
```

### WASI HTTP 서버 (컴포넌트 모델)

```rust
// wasi-http 사용 (실험적, 컴포넌트 모델)
// WASI가 나아가는 방향을 보여줌

use std::io::Write;

fn main() {
    // WASI는 다음에 대한 표준화된 인터페이스를 제공:
    // - 파일시스템 (wasi:filesystem)
    // - 소켓 (wasi:sockets)
    // - HTTP (wasi:http)
    // - 시계 (wasi:clocks)
    // - 난수 (wasi:random)

    println!("WASI는 이식 가능한 서버 사이드 Wasm을 가능하게 함");
    println!("wasmtime, wasmer, WasmEdge에서 동일한 바이너리 실행");
}
```

---

## 8. Yew 프레임워크

Yew는 React에서 영감을 받은 Rust 웹 프론트엔드 프레임워크입니다:

```toml
[dependencies]
yew = { version = "0.21", features = ["csr"] }
```

### 기본 컴포넌트

```rust
use yew::prelude::*;

#[function_component(App)]
fn app() -> Html {
    let counter = use_state(|| 0);

    let increment = {
        let counter = counter.clone();
        Callback::from(move |_| counter.set(*counter + 1))
    };

    let decrement = {
        let counter = counter.clone();
        Callback::from(move |_| counter.set(*counter - 1))
    };

    html! {
        <div class="app">
            <h1>{ "Yew 카운터" }</h1>
            <p>{ format!("카운트: {}", *counter) }</p>
            <button onclick={increment}>{ "+1" }</button>
            <button onclick={decrement}>{ "-1" }</button>
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
```

### Props와 State를 사용한 컴포넌트

```rust
use yew::prelude::*;

#[derive(Properties, PartialEq)]
struct TodoItemProps {
    text: String,
    done: bool,
    on_toggle: Callback<()>,
}

#[function_component(TodoItem)]
fn todo_item(props: &TodoItemProps) -> Html {
    let style = if props.done { "text-decoration: line-through" } else { "" };

    html! {
        <li style={style} onclick={props.on_toggle.reform(|_| ())}>
            { &props.text }
        </li>
    }
}

#[derive(Clone, PartialEq)]
struct Todo {
    text: String,
    done: bool,
}

#[function_component(TodoApp)]
fn todo_app() -> Html {
    let todos = use_state(|| vec![
        Todo { text: "Rust 배우기".into(), done: true },
        Todo { text: "Yew 배우기".into(), done: false },
        Todo { text: "무언가 만들기".into(), done: false },
    ]);

    let input_ref = use_node_ref();

    let on_add = {
        let todos = todos.clone();
        let input_ref = input_ref.clone();
        Callback::from(move |_| {
            if let Some(input) = input_ref.cast::<web_sys::HtmlInputElement>() {
                let text = input.value();
                if !text.is_empty() {
                    let mut new_todos = (*todos).clone();
                    new_todos.push(Todo { text, done: false });
                    todos.set(new_todos);
                    input.set_value("");
                }
            }
        })
    };

    html! {
        <div>
            <h1>{ "할 일 앱" }</h1>
            <div>
                <input ref={input_ref} placeholder="새 할 일..." />
                <button onclick={on_add}>{ "추가" }</button>
            </div>
            <ul>
                { for todos.iter().enumerate().map(|(i, todo)| {
                    let todos = todos.clone();
                    let on_toggle = Callback::from(move |_| {
                        let mut new_todos = (*todos).clone();
                        new_todos[i].done = !new_todos[i].done;
                        todos.set(new_todos);
                    });
                    html! {
                        <TodoItem
                            text={todo.text.clone()}
                            done={todo.done}
                            on_toggle={on_toggle}
                        />
                    }
                })}
            </ul>
        </div>
    }
}
```

Trunk으로 빌드:

```bash
cargo install trunk
trunk serve  # 핫 리로드가 있는 개발 서버
trunk build --release  # 프로덕션 빌드
```

---

## 9. 바이너리 크기 최적화

Wasm 바이너리 크기는 로드 시간에 직접 영향을 미칩니다:

```toml
# Cargo.toml
[profile.release]
opt-level = "z"         # 크기 최적화 (공격적)
lto = true              # 링크 타임 최적화
codegen-units = 1       # 단일 코드젠 유닛 (빌드 느림, 최적화 좋음)
strip = true            # 디버그 심볼 제거
panic = "abort"         # 언와인딩 코드 없음
```

### wasm-opt 후처리

```bash
# binaryen 도구 설치
# brew install binaryen (macOS)
# apt install binaryen (Ubuntu)

# Wasm 바이너리 추가 최적화
wasm-opt -Oz -o optimized.wasm original.wasm

# 일반적인 크기 감소 파이프라인:
# 1. Cargo 릴리즈 빌드:  ~200KB
# 2. wasm-opt:             ~150KB
# 3. gzip 압축:            ~50KB
```

### 크기 분석

```bash
# Wasm 바이너리에서 공간을 차지하는 것 분석
cargo install twiggy

twiggy top target/wasm32-unknown-unknown/release/my_lib.wasm
twiggy dominators target/wasm32-unknown-unknown/release/my_lib.wasm
```

### 일반적인 크기 감소 방법

```rust
// 1. 필요한 것에만 #[wasm_bindgen] 사용
// 2. 가능하면 format!과 println! 피하기 (포매팅 기계를 가져옴)
// 3. 라이브러리에는 가능하면 no_std 사용

// 4. 가능하면 String 대신 &str 사용
#[wasm_bindgen]
pub fn process(input: &str) -> String {  // &str 입력은 할당 방지
    input.to_uppercase()
}

// 5. 큰 의존성 가져오기 방지
// web-sys 기능 플래그로 필요한 것만 포함
```

---

## 10. Wasm 디버깅

### 콘솔 로깅

```rust
// Wasm용 간단한 로깅 매크로
macro_rules! console_log {
    ($($t:tt)*) => {
        web_sys::console::log_1(&format!($($t)*).into())
    };
}

#[wasm_bindgen]
pub fn debug_demo() {
    console_log!("디버그 값: {}", 42);
    console_log!("복잡한 값: {:?}", vec![1, 2, 3]);
}
```

### 패닉 훅

```rust
use wasm_bindgen::prelude::*;

// Rust 패닉을 스택 트레이스와 함께 콘솔 에러로 표시
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
```

```toml
[dependencies]
console_error_panic_hook = "0.1"
```

### 브라우저 개발자 도구

```
1. 디버그 정보로 빌드: wasm-pack build --dev
2. 브라우저 개발자 도구 → Sources 탭 열기
3. Rust 소스 맵으로 Rust 코드 단계별 실행 가능
4. Memory 탭에서 Wasm 선형 메모리 확인
```

### 테스팅

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_greet() {
        assert_eq!(greet("Rust"), "안녕하세요, Rust!");
    }

    #[wasm_bindgen_test]
    fn test_calculator() {
        let mut calc = Calculator::new();
        calc.add(10.0);
        calc.multiply(3.0);
        assert_eq!(calc.result(), 30.0);
    }
}
```

```bash
wasm-pack test --chrome --headless
```

---

## 11. 실용적 패턴

### JavaScript와 메모리 공유

```rust
use wasm_bindgen::prelude::*;

// 제로 카피 데이터 공유를 위한 원시 메모리 노출
#[wasm_bindgen]
pub struct ImageProcessor {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[wasm_bindgen]
impl ImageProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height * 4) as usize;  // RGBA
        ImageProcessor {
            width,
            height,
            pixels: vec![0; size],
        }
    }

    // 픽셀 버퍼에 대한 포인터 반환
    // JavaScript는 Wasm 메모리에 대한 Uint8Array 뷰를 만들 수 있음
    pub fn pixels_ptr(&self) -> *const u8 {
        self.pixels.as_ptr()
    }

    pub fn pixels_len(&self) -> usize {
        self.pixels.len()
    }

    // 이미지 처리 (예: 회색조 변환)
    pub fn grayscale(&mut self) {
        for chunk in self.pixels.chunks_exact_mut(4) {
            let gray = (0.299 * chunk[0] as f64
                      + 0.587 * chunk[1] as f64
                      + 0.114 * chunk[2] as f64) as u8;
            chunk[0] = gray;
            chunk[1] = gray;
            chunk[2] = gray;
            // chunk[3] (알파)는 변경 없음
        }
    }
}
```

JavaScript에서 사용:

```javascript
const processor = new ImageProcessor(800, 600);

// Wasm 메모리에 대한 뷰 얻기 (제로 카피!)
const pixels = new Uint8Array(
  wasm.memory.buffer,
  processor.pixels_ptr(),
  processor.pixels_len()
);

// 캔버스에서 이미지 데이터를 Wasm 메모리로 복사
const ctx = canvas.getContext('2d');
const imageData = ctx.getImageData(0, 0, 800, 600);
pixels.set(imageData.data);

// Rust에서 처리 (빠름!)
processor.grayscale();

// 캔버스로 다시 복사
imageData.data.set(pixels);
ctx.putImageData(imageData, 0, 0);
```

---

## 12. 연습문제

1. **Markdown 렌더러**: Markdown 텍스트를 HTML로 변환하는 Wasm 모듈을 빌드하세요. `pulldown-cmark` 크레이트를 사용하세요. textarea 입력과 실시간 미리보기가 있는 간단한 웹 페이지를 만드세요.

2. **Game of Life**: Conway's Game of Life를 Rust + Wasm으로 구현하세요. HTML 캔버스에 그리드를 렌더링하세요. 시작/중지/스텝 컨트롤을 추가하세요.

3. **JSON 포매터**: Wasm 기반 JSON 포매터/검증기를 빌드하세요. JSON 문자열을 입력받아 구문 강조와 함께 예쁘게 인쇄된 JSON을 출력하세요.

4. **WASI CLI 도구**: CSV 파일을 읽고, 열별 통계(평균, 중앙값, 표준편차)를 계산하고, 요약을 출력하는 WASI 커맨드라인 도구를 작성하세요. wasmtime으로 테스트하세요.

5. **Yew TODO 앱**: Yew로 완전한 TODO 애플리케이션을 빌드하세요: 항목 추가/제거/토글, 필터(전체/활성/완료), 로컬 스토리지 영속성, 키보드 단축키 포함.

---

## 참고 자료

- [Rust and WebAssembly Book](https://rustwasm.github.io/docs/book/)
- [wasm-bindgen Guide](https://rustwasm.github.io/wasm-bindgen/)
- [web-sys documentation](https://rustwasm.github.io/wasm-bindgen/api/web_sys/)
- [WASI documentation](https://wasi.dev/)
- [Yew documentation](https://yew.rs/)
- [Trunk documentation](https://trunkrs.dev/)

---

**이전**: [FFI와 상호운용](./08_FFI_and_Interop.md) | **다음**: [임베디드 Rust](./10_Embedded_Rust.md)
