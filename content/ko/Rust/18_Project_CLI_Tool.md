# 18. 프로젝트: CLI 도구

**이전**: [안전하지 않은 Rust](./17_Unsafe_Rust.md)

## 학습 목표

이 프로젝트를 완료하면 다음을 할 수 있습니다:

1. 여러 모듈로 구성된 실제 Rust CLI 애플리케이션 구조화하기
2. 디라이브 매크로(Derive Macro)를 사용하여 `clap`으로 커맨드라인 인수 파싱하기
3. `serde`와 `serde_json`으로 데이터 구조를 직렬화(Serialize) 및 역직렬화(Deserialize)하기
4. `anyhow` 크레이트를 사용하여 관용적(Idiomatic)으로 에러 처리하기
5. 파일 I/O, 데이터 모델링, 사용자 상호작용을 하나의 응집된 애플리케이션으로 결합하기

---

이것은 캡스톤(Capstone) 프로젝트입니다. 개별 개념을 배우는 대신, 완전한 **할 일(Todo)** 커맨드라인 애플리케이션을 처음부터 만들 것입니다. 이 도구를 통해 사용자는 작업을 추가하고, 나열하고, 완료로 표시하고, 제거할 수 있습니다 — 모두 디스크의 JSON 파일에 저장됩니다. 그 과정에서 거의 모든 프로덕션 Rust CLI 도구에 등장하는 패턴과 크레이트를 사용하게 됩니다.

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [프로젝트 설정](#2-프로젝트-설정)
3. [Clap으로 CLI 인수 파싱](#3-clap으로-cli-인수-파싱)
4. [데이터 모델](#4-데이터-모델)
5. [파일 영속성](#5-파일-영속성)
6. [명령어 구현](#6-명령어-구현)
7. [Anyhow로 에러 처리](#7-anyhow로-에러-처리)
8. [컬러 출력](#8-컬러-출력)
9. [모든 것을 합치기](#9-모든-것을-합치기)
10. [확장 아이디어](#10-확장-아이디어)

---

## 1. 프로젝트 개요

완성된 도구의 실행 모습입니다:

```
$ todo add "Buy groceries"
Added: Buy groceries (id: 1)

$ todo add "Write Rust project"
Added: Write Rust project (id: 2)

$ todo add "Call dentist"
Added: Call dentist (id: 3)

$ todo list
 ID  Status  Description
  1  [ ]     Buy groceries
  2  [ ]     Write Rust project
  3  [ ]     Call dentist

$ todo done 2
Completed: Write Rust project

$ todo list
 ID  Status  Description
  1  [ ]     Buy groceries
  2  [x]     Write Rust project
  3  [ ]     Call dentist

$ todo remove 1
Removed: Buy groceries

$ todo list
 ID  Status  Description
  2  [x]     Write Rust project
  3  [ ]     Call dentist
```

### 아키텍처(Architecture)

```
todo-cli/
├── Cargo.toml
└── src/
    ├── main.rs       ← 진입점: 인수 파싱, 명령어 디스패치(Dispatch)
    ├── cli.rs         ← clap 인수 정의
    ├── model.rs       ← Todo와 TodoList 데이터 구조
    ├── storage.rs     ← JSON 파일 읽기/쓰기
    └── commands.rs    ← 각 명령어의 비즈니스 로직(Business Logic)
```

---

## 2. 프로젝트 설정

프로젝트를 생성하고 의존성을 추가합니다:

```bash
cargo new todo-cli
cd todo-cli

cargo add clap --features derive     # 디라이브 매크로를 사용한 인수 파싱
cargo add serde --features derive    # 직렬화 프레임워크
cargo add serde_json                 # JSON 포맷 지원
cargo add anyhow                     # 인체공학적 에러 처리
cargo add colored                    # 터미널 색상 (선택사항)
```

`Cargo.toml`은 다음과 같아야 합니다:

```toml
[package]
name = "todo-cli"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
colored = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

---

## 3. Clap으로 CLI 인수 파싱

Clap의 디라이브 매크로(Derive Macro)를 사용하면 CLI 인터페이스를 일반 Rust 구조체와 열거형으로 정의할 수 있습니다. 라이브러리가 모든 파싱, 도움말 텍스트, 에러 메시지를 자동으로 생성합니다.

```rust
// src/cli.rs
use clap::{Parser, Subcommand};

/// A simple command-line todo manager.
///
/// Stores tasks in a JSON file and supports adding, listing,
/// completing, and removing items.
#[derive(Parser)]
#[command(name = "todo", version, about)]
pub struct Cli {
    /// Path to the todo storage file
    #[arg(short, long, default_value = "todos.json")]
    pub file: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Add a new todo item
    Add {
        /// Description of the task
        #[arg(required = true)]
        description: Vec<String>,
    },

    /// List all todo items
    List,

    /// Mark a todo item as completed
    Done {
        /// ID of the todo to complete
        id: u64,
    },

    /// Remove a todo item
    Remove {
        /// ID of the todo to remove
        id: u64,
    },
}
```

**왜 clap 디라이브를 사용하나요?** 인수 파서를 직접 작성하는 것은 지루하고 오류가 발생하기 쉽습니다. 디라이브 방식은 CLI 정의를 데이터 타입 가까이 유지하고, clap은 `--help` 출력, 잘못된 입력에 대한 에러 메시지, 셸 자동완성을 무료로 생성합니다.

---

## 4. 데이터 모델

데이터 모델은 단순합니다: `Todo`는 하나의 작업을 보유하고, `TodoList`는 컬렉션(Collection)을 관리합니다. `Serialize`와 `Deserialize`를 디라이브하여 serde가 이 타입들을 JSON으로 자동 변환할 수 있게 합니다.

```rust
// src/model.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Todo {
    pub id: u64,
    pub description: String,
    pub completed: bool,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct TodoList {
    // 다음에 할당할 ID — 단조 증가하여 삭제 후에도 ID가
    // 재사용되지 않도록 합니다
    next_id: u64,
    pub items: Vec<Todo>,
}

impl TodoList {
    pub fn new() -> Self {
        TodoList {
            next_id: 1,
            items: Vec::new(),
        }
    }

    /// 작업을 추가하고 할당된 ID를 반환합니다.
    pub fn add(&mut self, description: String) -> u64 {
        let id = self.next_id;
        self.items.push(Todo {
            id,
            description,
            completed: false,
        });
        // 사용 후 증가하여 다음 작업이 새로운 ID를 갖도록 합니다
        self.next_id += 1;
        id
    }

    /// 작업을 완료로 표시합니다. 찾은 경우 설명을 반환합니다.
    pub fn mark_done(&mut self, id: u64) -> Option<String> {
        // .iter_mut()는 내부에서 수정할 수 있도록 가변 참조를 제공합니다
        self.items.iter_mut().find(|t| t.id == id).map(|todo| {
            todo.completed = true;
            todo.description.clone()
        })
    }

    /// ID로 작업을 제거합니다. 찾은 경우 설명을 반환합니다.
    pub fn remove(&mut self, id: u64) -> Option<String> {
        // 먼저 위치를 찾은 다음 인덱스로 제거합니다.
        // .position()은 Option<usize>를 반환합니다.
        let pos = self.items.iter().position(|t| t.id == id)?;
        let todo = self.items.remove(pos);
        Some(todo.description)
    }

    /// 목록에 항목이 없으면 true를 반환합니다.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_find() {
        let mut list = TodoList::new();
        let id = list.add("Test task".into());
        assert_eq!(id, 1);
        assert_eq!(list.items.len(), 1);
        assert_eq!(list.items[0].description, "Test task");
        assert!(!list.items[0].completed);
    }

    #[test]
    fn mark_done_existing() {
        let mut list = TodoList::new();
        list.add("Task".into());
        let result = list.mark_done(1);
        assert_eq!(result, Some("Task".into()));
        assert!(list.items[0].completed);
    }

    #[test]
    fn mark_done_nonexistent() {
        let mut list = TodoList::new();
        assert_eq!(list.mark_done(99), None);
    }

    #[test]
    fn remove_existing() {
        let mut list = TodoList::new();
        list.add("Task".into());
        let result = list.remove(1);
        assert_eq!(result, Some("Task".into()));
        assert!(list.is_empty());
    }

    #[test]
    fn ids_never_reuse() {
        let mut list = TodoList::new();
        let id1 = list.add("First".into());
        let id2 = list.add("Second".into());
        list.remove(id1);
        let id3 = list.add("Third".into());
        // id3은 1이 아닌 3이어야 합니다 (ID는 재사용되지 않습니다)
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }
}
```

---

## 5. 파일 영속성

작업은 JSON 파일로 저장됩니다. 시작 시 파일을 읽고 매 수정 후 다시 씁니다. `anyhow` 크레이트는 커스텀 에러 타입을 정의하지 않고도 에러에 설명적인 메시지를 추가하는 `Context`를 제공합니다.

```rust
// src/storage.rs
use std::path::Path;
use anyhow::{Context, Result};

use crate::model::TodoList;

/// JSON 파일에서 TodoList를 로드합니다.
/// 파일이 없으면 빈 TodoList를 반환합니다.
pub fn load(path: &str) -> Result<TodoList> {
    let path = Path::new(path);

    if !path.exists() {
        // 아직 파일이 없습니다 — 새로 시작합니다. 첫 실행 시 예상되는 상태이므로
        // 에러로 처리하지 않습니다.
        return Ok(TodoList::new());
    }

    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;

    let list: TodoList = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse JSON from {}", path.display()))?;

    Ok(list)
}

/// TodoList를 JSON 파일로 저장합니다.
/// 사람이 읽기 쉽도록 프리티 프린팅(Pretty-Printing)을 사용합니다.
pub fn save(path: &str, list: &TodoList) -> Result<()> {
    let contents = serde_json::to_string_pretty(list)
        .context("Failed to serialize TodoList to JSON")?;

    std::fs::write(path, contents)
        .with_context(|| format!("Failed to write {}", path))?;

    Ok(())
}
```

생성되는 JSON 파일은 다음과 같습니다:

```json
{
  "next_id": 4,
  "items": [
    {
      "id": 1,
      "description": "Buy groceries",
      "completed": false
    },
    {
      "id": 2,
      "description": "Write Rust project",
      "completed": true
    }
  ]
}
```

---

## 6. 명령어 구현

각 명령어는 가변 `TodoList`를 받아 작업을 수행하고 피드백을 출력하는 함수입니다. 명령어를 CLI 파싱과 분리하면 코드를 테스트하기 쉬워집니다 — 커맨드라인 인수를 시뮬레이션하지 않고도 테스트에서 이 함수들을 호출할 수 있습니다.

```rust
// src/commands.rs
use colored::Colorize;
use crate::model::TodoList;

/// 새 할 일 항목을 추가합니다. 설명 부분은 공백으로 결합되므로
/// 사용자가 여러 단어 설명에 따옴표를 사용할 필요가 없습니다.
pub fn add(list: &mut TodoList, description: Vec<String>) {
    let desc = description.join(" ");
    let id = list.add(desc.clone());
    println!("{} {} (id: {})", "Added:".green().bold(), desc, id);
}

/// 모든 할 일 항목을 서식이 지정된 테이블로 표시합니다.
pub fn list_todos(list: &TodoList) {
    if list.is_empty() {
        println!("{}", "No todos yet. Add one with `todo add <description>`".dimmed());
        return;
    }

    // 헤더 출력
    println!(
        " {:>3}  {:6}  {}",
        "ID".bold(),
        "Status".bold(),
        "Description".bold()
    );

    for todo in &list.items {
        let status = if todo.completed {
            "[x]".green().to_string()
        } else {
            "[ ]".to_string()
        };

        let desc = if todo.completed {
            // 완료된 작업을 흐리게 하여 대기 중인 작업이 돋보이도록 합니다
            todo.description.dimmed().to_string()
        } else {
            todo.description.clone()
        };

        println!(" {:>3}  {:6}  {}", todo.id, status, desc);
    }
}

/// ID로 할 일을 완료로 표시합니다.
pub fn done(list: &mut TodoList, id: u64) {
    match list.mark_done(id) {
        Some(desc) => println!("{} {}", "Completed:".green().bold(), desc),
        None => println!("{} No todo with id {}", "Error:".red().bold(), id),
    }
}

/// ID로 할 일을 제거합니다.
pub fn remove(list: &mut TodoList, id: u64) {
    match list.remove(id) {
        Some(desc) => println!("{} {}", "Removed:".yellow().bold(), desc),
        None => println!("{} No todo with id {}", "Error:".red().bold(), id),
    }
}
```

---

## 7. Anyhow로 에러 처리

애플리케이션 코드(라이브러리 코드와 대조적으로)에서는 `anyhow::Result`가 표준 선택입니다. 이것은 에러 타입을 단일 `anyhow::Error`로 지워버리며:

- 자동으로 백트레이스(Backtrace)를 캡처합니다
- 사람이 읽기 쉬운 설명을 추가하는 `.context("msg")`를 지원합니다
- 모든 `std::error::Error` 타입에서 `?` 연산자와 함께 작동합니다
- 표시 시 원인의 전체 체인을 출력합니다

```rust
// 우리 애플리케이션에서의 에러 흐름:
//
//   storage::load() → 실패할 수 있음 (IO 에러, JSON 파싱 에러)
//        ↓
//   commands::*()   → 불가능 (출력을 직접 출력)
//        ↓
//   storage::save() → 실패할 수 있음 (IO 에러, JSON 직렬화 에러)
//        ↓
//   main()          → 모든 에러를 잡아 출력하고 코드 1로 종료

// 에러 출력 예시:
//   Error: Failed to parse JSON from todos.json
//
//   Caused by:
//       expected value at line 1 column 1
```

**라이브러리** 크레이트(다른 사람이 의존하는 코드)의 경우, 호출자가 특정 변형(Variant)에 매칭할 수 있도록 구조화된 에러 열거형을 정의하는 `thiserror`를 선호하세요. **애플리케이션** 크레이트(이것과 같은 바이너리)의 경우, `anyhow`가 더 간단하고 충분합니다.

---

## 8. 컬러 출력

`colored` 크레이트는 문자열 타입에 `.green()`, `.bold()`, `.dimmed()` 같은 메서드를 추가합니다. `NO_COLOR` 환경 변수를 존중하고 비TTY(non-TTY) 출력을 자동으로 감지합니다.

```rust
use colored::Colorize;

fn demo() {
    println!("{}", "Success!".green().bold());
    println!("{}", "Warning!".yellow());
    println!("{}", "Error!".red().bold());
    println!("{}", "Muted text".dimmed());

    // 스타일 체이닝(Chaining)
    println!("{}", "Important".red().bold().underline());
}
```

이 의존성을 추가하고 싶지 않다면, 컬러 호출을 일반 `println!`으로 교체하세요. 애플리케이션은 어느 쪽이든 동일하게 작동합니다.

---

## 9. 모든 것을 합치기

### 9.1 메인 진입점

```rust
// src/main.rs
mod cli;
mod commands;
mod model;
mod storage;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();

    // 기존 할 일 로드 (또는 새로 시작)
    let mut list = storage::load(&cli.file)?;

    // 적절한 명령어로 디스패치
    match cli.command {
        Command::Add { description } => commands::add(&mut list, description),
        Command::List => commands::list_todos(&list),
        Command::Done { id } => commands::done(&mut list, id),
        Command::Remove { id } => commands::remove(&mut list, id),
    }

    // 변경 사항을 디스크에 저장
    storage::save(&cli.file, &list)?;

    Ok(())
}
```

### 9.2 전체 파일 목록

참고를 위한 프로젝트의 모든 파일입니다. 여기까지 따라왔다면 위의 섹션에서 이미 각각을 가지고 있습니다.

**`src/cli.rs`** — CLI 인수 정의 (섹션 3)

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "todo", version, about = "A simple command-line todo manager")]
pub struct Cli {
    #[arg(short, long, default_value = "todos.json")]
    pub file: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Add a new todo item
    Add {
        #[arg(required = true)]
        description: Vec<String>,
    },
    /// List all todo items
    List,
    /// Mark a todo item as completed
    Done { id: u64 },
    /// Remove a todo item
    Remove { id: u64 },
}
```

**`src/model.rs`** — 데이터 구조 (섹션 4)

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Todo {
    pub id: u64,
    pub description: String,
    pub completed: bool,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct TodoList {
    next_id: u64,
    pub items: Vec<Todo>,
}

impl TodoList {
    pub fn new() -> Self {
        TodoList {
            next_id: 1,
            items: Vec::new(),
        }
    }

    pub fn add(&mut self, description: String) -> u64 {
        let id = self.next_id;
        self.items.push(Todo {
            id,
            description,
            completed: false,
        });
        self.next_id += 1;
        id
    }

    pub fn mark_done(&mut self, id: u64) -> Option<String> {
        self.items.iter_mut().find(|t| t.id == id).map(|todo| {
            todo.completed = true;
            todo.description.clone()
        })
    }

    pub fn remove(&mut self, id: u64) -> Option<String> {
        let pos = self.items.iter().position(|t| t.id == id)?;
        let todo = self.items.remove(pos);
        Some(todo.description)
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
```

**`src/storage.rs`** — 파일 영속성 (섹션 5)

```rust
use std::path::Path;
use anyhow::{Context, Result};
use crate::model::TodoList;

pub fn load(path: &str) -> Result<TodoList> {
    let path = Path::new(path);
    if !path.exists() {
        return Ok(TodoList::new());
    }
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let list: TodoList = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse JSON from {}", path.display()))?;
    Ok(list)
}

pub fn save(path: &str, list: &TodoList) -> Result<()> {
    let contents = serde_json::to_string_pretty(list)
        .context("Failed to serialize TodoList to JSON")?;
    std::fs::write(path, contents)
        .with_context(|| format!("Failed to write {}", path))?;
    Ok(())
}
```

**`src/commands.rs`** — 명령어 구현 (섹션 6)

```rust
use colored::Colorize;
use crate::model::TodoList;

pub fn add(list: &mut TodoList, description: Vec<String>) {
    let desc = description.join(" ");
    let id = list.add(desc.clone());
    println!("{} {} (id: {})", "Added:".green().bold(), desc, id);
}

pub fn list_todos(list: &TodoList) {
    if list.is_empty() {
        println!(
            "{}",
            "No todos yet. Add one with `todo add <description>`".dimmed()
        );
        return;
    }
    println!(
        " {:>3}  {:6}  {}",
        "ID".bold(),
        "Status".bold(),
        "Description".bold()
    );
    for todo in &list.items {
        let status = if todo.completed {
            "[x]".green().to_string()
        } else {
            "[ ]".to_string()
        };
        let desc = if todo.completed {
            todo.description.dimmed().to_string()
        } else {
            todo.description.clone()
        };
        println!(" {:>3}  {:6}  {}", todo.id, status, desc);
    }
}

pub fn done(list: &mut TodoList, id: u64) {
    match list.mark_done(id) {
        Some(desc) => println!("{} {}", "Completed:".green().bold(), desc),
        None => println!("{} No todo with id {}", "Error:".red().bold(), id),
    }
}

pub fn remove(list: &mut TodoList, id: u64) {
    match list.remove(id) {
        Some(desc) => println!("{} {}", "Removed:".yellow().bold(), desc),
        None => println!("{} No todo with id {}", "Error:".red().bold(), id),
    }
}
```

**`src/main.rs`** — 진입점

```rust
mod cli;
mod commands;
mod model;
mod storage;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut list = storage::load(&cli.file)?;

    match cli.command {
        Command::Add { description } => commands::add(&mut list, description),
        Command::List => commands::list_todos(&list),
        Command::Done { id } => commands::done(&mut list, id),
        Command::Remove { id } => commands::remove(&mut list, id),
    }

    storage::save(&cli.file, &list)?;
    Ok(())
}
```

### 9.3 빌드 및 테스트

```bash
# 프로젝트 빌드
cargo build

# 테스트 실행 (섹션 4의 model 테스트)
cargo test

# 사용해보기
cargo run -- add "Learn Rust"
cargo run -- add "Build a CLI tool"
cargo run -- list
cargo run -- done 1
cargo run -- list
cargo run -- remove 2
cargo run -- list

# 커스텀 파일 경로 사용
cargo run -- --file work.json add "Ship feature"

# 도움말 보기
cargo run -- --help
cargo run -- add --help

# 릴리즈 바이너리 빌드 및 설치
cargo build --release
# 바이너리는 target/release/todo-cli에 있습니다
# 전역적으로 사용하려면 PATH의 디렉토리로 복사하세요
```

---

## 10. 확장 아이디어

기본 도구가 작동하면, 다음 기능들을 추가하여 Rust 실력을 심화해보세요:

**마감일** — `Todo`에 `Option<String>` (또는 `chrono` 크레이트를 사용하는 `Option<NaiveDate>`)를 추가합니다. `add` 명령어에 `--due` 플래그를 추가합니다. 기한이 지난 작업을 빨간색으로 강조합니다.

**우선순위 수준** — `Todo`에 `Priority` 열거형 (`Low`, `Medium`, `High`)을 추가합니다. 목록 표시를 우선순위별로 정렬합니다. `add` 명령어에 `--priority`를 추가합니다.

**검색 및 필터** — 부분 문자열 매칭으로 할 일을 필터링하는 `search` 서브커맨드를 추가합니다. 완료되거나 대기 중인 항목만 표시하는 `--done` / `--pending` 플래그를 `list`에 추가합니다.

**내보내기 형식** — CSV, 마크다운 테이블, 또는 일반 텍스트 형식으로 출력하는 `export` 서브커맨드를 추가합니다. `--format` 플래그로 선택합니다.

**여러 목록** — 이름 있는 목록(`todo --list work add "Ship feature"`)을 지원합니다. 별도의 JSON 파일로 저장하거나 최상위 맵이 있는 단일 파일로 저장합니다.

**인터랙티브 모드** — `ratatui` 크레이트를 사용하여 인터랙티브 터미널 UI를 시작하는 `tui` 피처를 추가합니다. 사용자는 화살표 키로 탐색하고 Enter로 완료를 토글합니다.

**실행 취소** — 작업 기록을 유지하고 마지막 변경을 되돌리는 `todo undo`를 지원합니다. 항목과 함께 JSON 파일에 기록을 저장합니다.

---

## 참고 자료
- [clap 문서](https://docs.rs/clap/latest/clap/)
- [serde 문서](https://serde.rs/)
- [anyhow 문서](https://docs.rs/anyhow/latest/anyhow/)
- [colored 문서](https://docs.rs/colored/latest/colored/)
- [Command Line Applications in Rust (도서)](https://rust-cli.github.io/book/)

---

**이전**: [안전하지 않은 Rust](./17_Unsafe_Rust.md)
