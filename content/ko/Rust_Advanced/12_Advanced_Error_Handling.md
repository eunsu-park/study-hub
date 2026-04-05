# 28. 고급 에러 처리

**이전**: [네트워크 프로그래밍](./11_Network_Programming.md) | **다음**: [성능과 프로파일링](./13_Performance_Profiling.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 라이브러리와 애플리케이션을 위한 커스텀 에러 타입 계층 설계하기
2. 라이브러리 에러 타입에 `thiserror`, 애플리케이션 에러 처리에 `anyhow` 사용하기
3. 모듈 경계를 넘어 변환 트레이트로 에러 합성하기
4. 재시도, 폴백, 서킷 브레이커 등의 복구 패턴 적용하기
5. 백트레이스, 소스 체인, 스팬 정보로 풍부한 에러 컨텍스트 제공하기

---

레슨 09에서 Rust의 `Result`/`Option` 기초와 `?` 연산자를 다뤘습니다. 이 레슨은 대규모 에러 처리를 다룹니다: 실제 애플리케이션을 위한 에러 타입 설계, 상황에 맞는 크레이트 선택, 레이어 간 에러 합성, 프로덕션에서의 우아한 에러 처리.

## 목차
1. [에러 처리 철학](#1-에러-처리-철학)
2. [처음부터 커스텀 에러 타입](#2-처음부터-커스텀-에러-타입)
3. [thiserror: 라이브러리 에러 타입](#3-thiserror-라이브러리-에러-타입)
4. [anyhow: 애플리케이션 에러 처리](#4-anyhow-애플리케이션-에러-처리)
5. [레이어 간 에러 합성](#5-레이어-간-에러-합성)
6. [에러 컨텍스트와 백트레이스](#6-에러-컨텍스트와-백트레이스)
7. [복구 패턴](#7-복구-패턴)
8. [비동기 코드의 에러 처리](#8-비동기-코드의-에러-처리)
9. [로깅과 보고](#9-로깅과-보고)
10. [에러 경로 테스팅](#10-에러-경로-테스팅)
11. [설계 가이드라인](#11-설계-가이드라인)
12. [연습문제](#12-연습문제)

---

## 1. 에러 처리 철학

### 라이브러리 vs 애플리케이션 코드

| 컨텍스트 | 접근법 | 크레이트 | 이유 |
|---------|--------|---------|------|
| **라이브러리** | 타입화된 에러 | `thiserror` | 호출자가 특정 에러 변형을 매칭해야 함 |
| **애플리케이션** | 컨텍스트 에러 | `anyhow` | 최상위 코드는 주로 에러를 로깅/보고 |
| **둘 다** | 중요한 곳에 타입 사용 | 혼합 | 공개 API는 타입 사용, 내부는 `anyhow` 사용 |

### Error 트레이트

모든 Rust 에러는 `std::error::Error`를 구현합니다:

```rust
pub trait Error: Display + Debug {
    fn source(&self) -> Option<&(dyn Error + 'static)> { None }
    // 더 이상 사용되지 않음: fn description()과 fn cause()
}
```

체인: 사람이 읽을 수 있는 메시지를 위한 `Display`, 개발자 진단을 위한 `Debug`, 에러 연결을 위한 `source()`.

---

## 2. 처음부터 커스텀 에러 타입

### 열거형 기반 에러 타입

```rust
use std::fmt;
use std::io;
use std::num::ParseIntError;

#[derive(Debug)]
pub enum AppError {
    Io(io::Error),
    Parse(ParseIntError),
    Config { key: String, message: String },
    NotFound(String),
    Unauthorized,
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "I/O 에러: {e}"),
            AppError::Parse(e) => write!(f, "파싱 에러: {e}"),
            AppError::Config { key, message } => {
                write!(f, "'{key}' 설정 에러: {message}")
            }
            AppError::NotFound(item) => write!(f, "찾을 수 없음: {item}"),
            AppError::Unauthorized => write!(f, "인증 실패"),
        }
    }
}

impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AppError::Io(e) => Some(e),
            AppError::Parse(e) => Some(e),
            _ => None,
        }
    }
}

// ?로 자동 변환을 위한 From 구현
impl From<io::Error> for AppError {
    fn from(e: io::Error) -> Self {
        AppError::Io(e)
    }
}

impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self {
        AppError::Parse(e)
    }
}

// 사용 예
fn load_config(path: &str) -> Result<u32, AppError> {
    let content = std::fs::read_to_string(path)?;  // io::Error → AppError::Io
    let value: u32 = content.trim().parse()?;       // ParseIntError → AppError::Parse
    Ok(value)
}
```

보일러플레이트가 많습니다. `thiserror`가 대부분을 제거해 줍니다.

---

## 3. thiserror: 라이브러리 에러 타입

`thiserror`는 `Display`, `Error`, `From` 구현을 파생(derive)합니다:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("I/O 에러: {0}")]
    Io(#[from] std::io::Error),

    #[error("라인 {line}에서 파싱 에러: {message}")]
    Parse { line: usize, message: String },

    #[error("검증 실패: {0}")]
    Validation(String),

    #[error("레코드를 찾을 수 없음: {id}")]
    NotFound { id: u64 },

    #[error("사용자 {user}가 리소스 {resource}에 접근 거부됨")]
    Forbidden { user: String, resource: String },

    #[error(transparent)]  // 내부 에러에 Display를 위임
    Other(#[from] anyhow::Error),
}

// #[from] 속성은 DataError를 위한 From<io::Error>를 생성
// #[error("...")] 속성은 Display 구현을 생성

fn read_record(path: &str, id: u64) -> Result<String, DataError> {
    let content = std::fs::read_to_string(path)?;  // 자동 변환

    let record = content
        .lines()
        .find(|line| line.starts_with(&id.to_string()))
        .ok_or(DataError::NotFound { id })?;

    if record.contains("RESTRICTED") {
        return Err(DataError::Forbidden {
            user: "anonymous".into(),
            resource: format!("record:{id}"),
        });
    }

    Ok(record.to_string())
}
```

### 구조체 에러 타입

```rust
use thiserror::Error;

#[derive(Debug, Error)]
#[error("{host}:{port}에 {attempts}번 시도 후 연결 실패")]
pub struct ConnectionError {
    pub host: String,
    pub port: u16,
    pub attempts: u32,
    #[source]
    pub cause: Option<std::io::Error>,
}

fn connect(host: &str, port: u16) -> Result<(), ConnectionError> {
    Err(ConnectionError {
        host: host.to_string(),
        port,
        attempts: 3,
        cause: Some(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "connection refused",
        )),
    })
}
```

---

## 4. anyhow: 애플리케이션 에러 처리

`anyhow::Error`는 모든 에러를 래핑하고 컨텍스트를 추가합니다. 애플리케이션 코드에 적합합니다:

```rust
use anyhow::{Context, Result, bail, ensure};

fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("설정 파일 읽기 실패: {path}"))?;

    let config: Config = toml::from_str(&content)
        .with_context(|| format!("설정 파일 파싱 실패: {path}"))?;

    ensure!(config.port > 0, "포트는 양수여야 합니다, 받은 값: {}", config.port);

    if config.host.is_empty() {
        bail!("호스트는 비어있을 수 없습니다");
    }

    Ok(config)
}

#[derive(serde::Deserialize)]
struct Config {
    host: String,
    port: u16,
}

fn main() -> Result<()> {
    let config = load_config("config.toml")?;
    println!("설정 로드됨: {}:{}", config.host, config.port);
    Ok(())
}
```

### anyhow 기능

```rust
use anyhow::{anyhow, Context, Result};

fn process() -> Result<()> {
    // 즉석 에러 생성
    let _ = Err(anyhow!("문제가 발생했습니다"))?;

    // 기존 에러에 컨텍스트 추가
    let data = std::fs::read("data.bin")
        .context("데이터 파일 로드 실패")?;

    // 여러 컨텍스트 연결
    let parsed = parse_data(&data)
        .context("데이터 파싱 실패")
        .context("입력 처리 중")?;

    // 특정 에러를 확인하기 위한 다운캐스트
    if let Err(e) = might_fail() {
        if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
            if io_err.kind() == std::io::ErrorKind::NotFound {
                println!("파일을 찾을 수 없습니다, 기본값 사용");
                return Ok(());
            }
        }
        return Err(e);
    }

    Ok(())
}

fn parse_data(data: &[u8]) -> Result<Vec<u8>> {
    Ok(data.to_vec())
}

fn might_fail() -> Result<()> {
    Ok(())
}
```

---

## 5. 레이어 간 에러 합성

### 레이어 아키텍처

```rust
// 도메인 레이어 — 타입화된 에러
mod domain {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum DomainError {
        #[error("사용자를 찾을 수 없음: {0}")]
        UserNotFound(u64),
        #[error("잘못된 이메일: {0}")]
        InvalidEmail(String),
        #[error("중복된 사용자명: {0}")]
        DuplicateUsername(String),
    }
}

// 인프라 레이어 — 외부 에러를 래핑
mod infra {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum InfraError {
        #[error("데이터베이스 에러: {0}")]
        Database(#[from] sqlx::Error),
        #[error("캐시 에러: {0}")]
        Cache(String),
        #[error("외부 API 에러: {0}")]
        ExternalApi(#[from] reqwest::Error),
    }
}

// 서비스 레이어 — 도메인과 인프라 에러를 합성
mod service {
    use thiserror::Error;
    use super::{domain, infra};

    #[derive(Debug, Error)]
    pub enum ServiceError {
        #[error(transparent)]
        Domain(#[from] domain::DomainError),
        #[error(transparent)]
        Infra(#[from] infra::InfraError),
        #[error("서비스 사용 불가: {0}")]
        Unavailable(String),
    }
}

// HTTP 레이어 — 서비스 에러를 상태 코드로 변환
mod http {
    use super::service::ServiceError;
    use super::domain::DomainError;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};

    impl IntoResponse for ServiceError {
        fn into_response(self) -> Response {
            let (status, message) = match &self {
                ServiceError::Domain(DomainError::UserNotFound(_)) =>
                    (StatusCode::NOT_FOUND, self.to_string()),
                ServiceError::Domain(DomainError::InvalidEmail(_)) =>
                    (StatusCode::BAD_REQUEST, self.to_string()),
                ServiceError::Domain(DomainError::DuplicateUsername(_)) =>
                    (StatusCode::CONFLICT, self.to_string()),
                ServiceError::Infra(_) =>
                    (StatusCode::INTERNAL_SERVER_ERROR, "내부 에러".into()),
                ServiceError::Unavailable(_) =>
                    (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            };

            (status, message).into_response()
        }
    }
}
```

---

## 6. 에러 컨텍스트와 백트레이스

### 에러 체인 구성

```rust
use anyhow::{Context, Result};

fn read_user_settings(user_id: u64) -> Result<Settings> {
    let path = format!("/users/{user_id}/settings.json");

    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("사용자 {user_id}의 설정 읽기 실패"))?;

    let settings: Settings = serde_json::from_str(&content)
        .with_context(|| format!("{path}의 설정 파싱 실패"))?;

    validate_settings(&settings)
        .with_context(|| format!("사용자 {user_id}의 잘못된 설정"))?;

    Ok(settings)
}

// 에러 출력 예시 (체인 포함):
// Error: 사용자 42의 잘못된 설정
//
// Caused by:
//     0: /users/42/settings.json의 설정 파싱 실패
//     1: line 5 column 3에서 값 예상

#[derive(serde::Deserialize)]
struct Settings {
    theme: String,
}

fn validate_settings(s: &Settings) -> Result<()> {
    Ok(())
}
```

### 백트레이스

```rust
use std::backtrace::Backtrace;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("치명적 오류: {message}")]
pub struct CriticalError {
    pub message: String,
    pub backtrace: Backtrace,
}

impl CriticalError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            backtrace: Backtrace::capture(),
        }
    }
}

// anyhow는 RUST_BACKTRACE=1일 때 자동으로 백트레이스를 캡처
// anyhow::Error는 .backtrace() 메서드를 가짐
```

---

## 7. 복구 패턴

### 지수 백오프 재시도

```rust
use std::time::Duration;
use tokio::time::sleep;

pub async fn retry<F, Fut, T, E>(
    max_attempts: u32,
    initial_delay: Duration,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut delay = initial_delay;
    let mut last_error = None;

    for attempt in 1..=max_attempts {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                eprintln!("시도 {attempt}/{max_attempts} 실패: {e}");
                last_error = Some(e);

                if attempt < max_attempts {
                    sleep(delay).await;
                    delay *= 2;  // 지수 백오프
                }
            }
        }
    }

    Err(last_error.unwrap())
}

// 사용법:
// let result = retry(3, Duration::from_millis(100), || async {
//     connect_to_database().await
// }).await?;
```

### 폴백 체인

```rust
async fn get_user_data(user_id: u64) -> Result<UserData, anyhow::Error> {
    // 먼저 캐시 시도
    if let Ok(data) = cache_lookup(user_id).await {
        return Ok(data);
    }

    // 데이터베이스로 폴백
    if let Ok(data) = db_lookup(user_id).await {
        // 다음 번을 위해 캐시에 저장
        let _ = cache_store(user_id, &data).await;
        return Ok(data);
    }

    // 기본값으로 폴백
    Ok(UserData::default_for(user_id))
}

struct UserData {
    id: u64,
    name: String,
}

impl UserData {
    fn default_for(id: u64) -> Self {
        UserData { id, name: format!("User {id}") }
    }
}

async fn cache_lookup(id: u64) -> Result<UserData, anyhow::Error> {
    anyhow::bail!("캐시 미스")
}

async fn db_lookup(id: u64) -> Result<UserData, anyhow::Error> {
    Ok(UserData { id, name: "Alice".into() })
}

async fn cache_store(id: u64, data: &UserData) -> Result<(), anyhow::Error> {
    Ok(())
}
```

### 서킷 브레이커 (Circuit Breaker)

```rust
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub struct CircuitBreaker {
    failure_count: AtomicU32,
    threshold: u32,
    reset_timeout: Duration,
    last_failure: std::sync::Mutex<Option<Instant>>,
}

#[derive(Debug, PartialEq)]
pub enum CircuitState {
    Closed,     // 정상 동작
    Open,       // 실패 중, 요청 거절
    HalfOpen,   // 서비스 복구 테스트 중
}

impl CircuitBreaker {
    pub fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_count: AtomicU32::new(0),
            threshold,
            reset_timeout,
            last_failure: std::sync::Mutex::new(None),
        }
    }

    pub fn state(&self) -> CircuitState {
        let failures = self.failure_count.load(Ordering::Relaxed);
        if failures < self.threshold {
            return CircuitState::Closed;
        }

        let last = self.last_failure.lock().unwrap();
        if let Some(last_time) = *last {
            if last_time.elapsed() > self.reset_timeout {
                CircuitState::HalfOpen
            } else {
                CircuitState::Open
            }
        } else {
            CircuitState::Closed
        }
    }

    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        *self.last_failure.lock().unwrap() = None;
    }

    pub fn record_failure(&self) {
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        *self.last_failure.lock().unwrap() = Some(Instant::now());
    }

    pub async fn call<F, Fut, T, E>(&self, operation: F) -> Result<T, CircuitError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
    {
        match self.state() {
            CircuitState::Open => Err(CircuitError::Open),
            _ => {
                match operation().await {
                    Ok(value) => {
                        self.record_success();
                        Ok(value)
                    }
                    Err(e) => {
                        self.record_failure();
                        Err(CircuitError::Inner(e))
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum CircuitError<E> {
    Open,
    Inner(E),
}
```

---

## 8. 비동기 코드의 에러 처리

### 스폰된 태스크의 에러 처리

```rust
use tokio::task::JoinError;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // JoinHandle은 Result<T, JoinError>를 반환
    // JoinError는 태스크가 패닉하거나 취소된 경우
    let handle = tokio::spawn(async {
        might_fail().await
    });

    match handle.await {
        Ok(Ok(value)) => println!("성공: {value}"),
        Ok(Err(e)) => println!("태스크가 에러 반환: {e}"),
        Err(join_err) => {
            if join_err.is_panic() {
                println!("태스크 패닉!");
            } else {
                println!("태스크가 취소됨");
            }
        }
    }

    Ok(())
}

async fn might_fail() -> anyhow::Result<String> {
    Ok("완료".into())
}
```

### 여러 태스크에서 결과 수집

```rust
use futures_util::future::join_all;

async fn fetch_all(urls: &[&str]) -> Vec<anyhow::Result<String>> {
    let futures: Vec<_> = urls.iter().map(|url| async move {
        let resp = reqwest::get(*url).await?;
        let body = resp.text().await?;
        Ok(body)
    }).collect();

    join_all(futures).await
}

// 성공과 실패로 분류
async fn fetch_with_report(urls: &[&str]) {
    let results = fetch_all(urls).await;

    let (successes, failures): (Vec<_>, Vec<_>) = results
        .into_iter()
        .enumerate()
        .partition(|(_, r)| r.is_ok());

    println!("{} 성공, {} 실패", successes.len(), failures.len());

    for (i, result) in failures {
        if let Err(e) = result {
            eprintln!("URL[{i}] 실패: {e:#}");
        }
    }
}
```

---

## 9. 로깅과 보고

### tracing을 이용한 구조화된 에러 로깅

```rust
use tracing::{error, warn, info, instrument};
use anyhow::{Context, Result};

#[instrument(skip(password))]
async fn authenticate(username: &str, password: &str) -> Result<User> {
    let user = find_user(username).await
        .with_context(|| format!("사용자 '{username}' 인증 실패"))?;

    if !verify_password(&user, password) {
        warn!(username, "잘못된 비밀번호 시도");
        anyhow::bail!("잘못된 자격 증명");
    }

    info!(username, user_id = user.id, "사용자 인증됨");
    Ok(user)
}

struct User { id: u64 }

async fn find_user(username: &str) -> Result<User> {
    Ok(User { id: 1 })
}

fn verify_password(user: &User, password: &str) -> bool {
    true
}
```

### 최종 사용자를 위한 에러 보고

```rust
fn report_error(error: &anyhow::Error) {
    // 최종 사용자용: 최상위 메시지만
    eprintln!("오류: {error}");

    // 개발자용: 전체 에러 체인
    if std::env::var("RUST_LOG").is_ok() {
        eprintln!("\n에러 체인:");
        for (i, cause) in error.chain().enumerate() {
            eprintln!("  {i}: {cause}");
        }

        // 가능한 경우 백트레이스
        let bt = error.backtrace();
        if bt.status() == std::backtrace::BacktraceStatus::Captured {
            eprintln!("\n백트레이스:\n{bt}");
        }
    }
}
```

---

## 10. 에러 경로 테스팅

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_email() {
        let result = validate_email("not-an-email");
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, DomainError::InvalidEmail(_)));
        assert!(err.to_string().contains("not-an-email"));
    }

    #[test]
    fn test_error_chain() {
        let result = load_config("/nonexistent/path");
        let err = result.unwrap_err();

        // 에러 체인 확인
        let mut chain = err.chain();
        assert!(chain.next().unwrap().to_string().contains("config"));
        assert!(chain.next().unwrap().to_string().contains("No such file"));
    }

    #[test]
    fn test_error_downcast() {
        let result: Result<(), anyhow::Error> = Err(
            DomainError::UserNotFound(42).into()
        );

        let err = result.unwrap_err();
        assert!(err.downcast_ref::<DomainError>().is_some());

        match err.downcast_ref::<DomainError>() {
            Some(DomainError::UserNotFound(id)) => assert_eq!(*id, 42),
            _ => panic!("UserNotFound 예상"),
        }
    }

    // 에러 메시지가 유용한지 테스트
    #[test]
    fn test_error_messages_are_descriptive() {
        let err = DomainError::InvalidEmail("bad".into());
        let msg = err.to_string();
        assert!(msg.contains("bad"), "에러는 잘못된 값을 언급해야 함");
        assert!(msg.to_lowercase().contains("email"), "에러는 무엇이 잘못됐는지 언급해야 함");
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
enum DomainError {
    #[error("잘못된 이메일: {0}")]
    InvalidEmail(String),
    #[error("사용자를 찾을 수 없음: {0}")]
    UserNotFound(u64),
}

fn validate_email(email: &str) -> Result<(), DomainError> {
    if !email.contains('@') {
        return Err(DomainError::InvalidEmail(email.to_string()));
    }
    Ok(())
}

fn load_config(path: &str) -> anyhow::Result<()> {
    std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!(e))
        .context("설정 로드 실패")?;
    Ok(())
}
```

---

## 11. 설계 가이드라인

### 해야 할 것

1. **라이브러리 에러에 구체적으로**: 설명적 변형의 열거형 사용
2. **전파 시 컨텍스트 추가**: `.context()` 또는 `.with_context()` 사용
3. **에러에 관련 데이터 포함**: ID, 경로, 실패 원인 값
4. **`source()` 구현**: 에러 체인 탐색 가능하게 하기
5. **에러 경로 테스트**: 행복한 경로만큼 중요

### 하지 말아야 할 것

1. **`String`을 에러 타입으로 사용하지 마세요**: 타입 안전성 없음, 합성 불가
2. **라이브러리 코드에서 `.unwrap()` 사용하지 마세요**: 대신 `Result` 반환
3. **에러 컨텍스트를 버리지 마세요**: `map_err(|_| MyError)`는 원인을 잃음
4. **복구 가능한 에러에 패닉하지 마세요**: `panic!`은 프로그래머 버그에만 사용
5. **같은 에러를 로깅하고 반환하지 마세요**: 하나만 선택, 아니면 호출자가 두 번 로깅

### 의사결정 트리

```
라이브러리 코드인가, 애플리케이션 코드인가?
├── 라이브러리 → thiserror 사용, 타입화된 에러 반환
├── 애플리케이션 → anyhow 사용, 컨텍스트 추가
└── 둘 다 → 공개 API는 thiserror, 내부는 anyhow
```

---

## 12. 연습문제

1. **에러 타입 계층**: 파일 처리 라이브러리를 위한 완전한 에러 타입 계층을 설계하세요: I/O, 파싱(라인/컬럼 정보 포함), 유효성 검증, 인코딩 에러. 적절한 `source()` 체인을 포함하세요.

2. **재시도 미들웨어**: 다음을 지원하는 제네릭 비동기 재시도 함수를 빌드하세요: 구성 가능한 최대 시도 횟수, 지터(jitter)가 있는 지수 백오프, 재시도 가능한 에러를 위한 조건자.

3. **서킷 브레이커**: 서킷 브레이커 패턴을 확장하여 지원하세요: 닫기 위한 구성 가능한 성공 임계값, 슬라이딩 윈도우 실패율, 비동기 함수와의 통합.

4. **에러 집계**: N개의 비동기 작업을 동시에 실행하고 결합된 결과를 반환하는 함수를 작성하세요: 모든 성공 또는 개별 에러가 포함된 실패 보고서.

5. **사용자 친화적 에러**: `anyhow::Error`를 받아 사용자 친화적 메시지(내부 세부 사항 없음)와 개발자 친화적 보고서(전체 체인 + 백트레이스)를 모두 생성하는 에러 리포터를 빌드하세요.

---

## 참고 자료

- [thiserror documentation](https://docs.rs/thiserror/latest/thiserror/)
- [anyhow documentation](https://docs.rs/anyhow/latest/anyhow/)
- [Error Handling in Rust (Rust Book)](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- [Jane Lusby: Error Handling in Rust](https://www.youtube.com/watch?v=rAF8mLI0naQ)

---

**이전**: [네트워크 프로그래밍](./11_Network_Programming.md) | **다음**: [성능과 프로파일링](./13_Performance_Profiling.md)
