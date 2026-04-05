# 30. 캡스톤: HTTP 서버

**이전**: [성능과 프로파일링](./13_Performance_Profiling.md)

## 학습 목표

이 프로젝트를 완료하면 다음을 할 수 있습니다:

1. 라우팅, 미들웨어, 에러 처리가 있는 완전한 HTTP 서버 설계하기
2. 유효성 검증과 페이지네이션이 있는 JSON API 엔드포인트 구현하기
3. 커넥션 풀링과 마이그레이션으로 데이터베이스 통합하기
4. JWT 토큰과 비밀번호 해싱으로 인증 추가하기
5. 프로덕션 배포를 위한 애플리케이션 준비하기

---

이것은 캡스톤 프로젝트입니다. 개별 개념을 배우는 대신, 완전한 **REST API 서버**를 처음부터 만들 것입니다. 이 프로젝트는 과정의 거의 모든 개념을 결합합니다: 소유권, 트레이트, 비동기, 에러 처리, 매크로, 네트워킹. 사용자, 게시물, 댓글, 인증, 데이터베이스 저장이 있는 프로덕션 품질의 블로그 API를 구축합니다.

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [프로젝트 설정](#2-프로젝트-설정)
3. [데이터베이스 스키마와 마이그레이션](#3-데이터베이스-스키마와-마이그레이션)
4. [애플리케이션 아키텍처](#4-애플리케이션-아키텍처)
5. [데이터베이스 모델](#5-데이터베이스-모델)
6. [에러 처리 레이어](#6-에러-처리-레이어)
7. [인증](#7-인증)
8. [API 라우트: 사용자](#8-api-라우트-사용자)
9. [API 라우트: 게시물](#9-api-라우트-게시물)
10. [API 라우트: 댓글](#10-api-라우트-댓글)
11. [미들웨어 스택](#11-미들웨어-스택)
12. [테스팅](#12-테스팅)
13. [배포](#13-배포)
14. [확장 아이디어](#14-확장-아이디어)

---

## 1. 프로젝트 개요

완성된 API의 모습입니다:

```
POST   /api/auth/register     — 계정 생성
POST   /api/auth/login        — JWT 토큰 획득
GET    /api/users/:id         — 사용자 프로필 조회
GET    /api/posts             — 게시물 목록 (페이지네이션)
POST   /api/posts             — 게시물 작성 (인증 필요)
GET    /api/posts/:id         — 댓글과 함께 게시물 조회
PUT    /api/posts/:id         — 게시물 수정 (작성자만)
DELETE /api/posts/:id         — 게시물 삭제 (작성자만)
POST   /api/posts/:id/comments — 댓글 추가 (인증 필요)
GET    /health                — 상태 확인
```

### 기술 스택

| 컴포넌트 | 선택 | 이유 |
|---------|------|------|
| HTTP | Axum | 인체공학적, Tower 기반, 가장 빠른 Rust 웹 프레임워크 |
| 데이터베이스 | SQLite + SQLx | 무설정, 비동기, 컴파일 타임 쿼리 검사 |
| 인증 | JWT (jsonwebtoken) | 무상태, 표준 |
| 비밀번호 | Argon2 (argon2) | OWASP 권장 |
| 유효성 검증 | validator | 파생(derive) 기반 구조체 유효성 검증 |
| 에러 처리 | thiserror + anyhow | 타입화된 API 에러, 컨텍스트 내부 |
| 로깅 | tracing | 구조화된, 비동기 인식 |
| 테스팅 | axum-test | 인프로세스 HTTP 테스팅 |

### 아키텍처

```
blog-api/
├── Cargo.toml
├── migrations/
│   └── 001_initial.sql
├── src/
│   ├── main.rs           ← 진입점, 서버 시작
│   ├── config.rs          ← 환경에서 설정 로드
│   ├── db.rs              ← 데이터베이스 풀 설정
│   ├── error.rs           ← 에러 타입과 변환
│   ├── auth/
│   │   ├── mod.rs         ← 인증 모듈 루트
│   │   ├── jwt.rs         ← JWT 토큰 생성/검증
│   │   ├── password.rs    ← 비밀번호 해싱
│   │   └── middleware.rs  ← 인증 추출 미들웨어
│   ├── models/
│   │   ├── mod.rs
│   │   ├── user.rs        ← 사용자 쿼리
│   │   ├── post.rs        ← 게시물 쿼리
│   │   └── comment.rs     ← 댓글 쿼리
│   ├── routes/
│   │   ├── mod.rs         ← 라우터 조립
│   │   ├── auth.rs        ← /api/auth/* 핸들러
│   │   ├── users.rs       ← /api/users/* 핸들러
│   │   ├── posts.rs       ← /api/posts/* 핸들러
│   │   └── comments.rs    ← /api/posts/*/comments 핸들러
│   └── middleware.rs      ← 커스텀 미들웨어 (타이밍, 로깅)
└── tests/
    ├── auth_test.rs
    ├── posts_test.rs
    └── common/mod.rs      ← 테스트 헬퍼
```

---

## 2. 프로젝트 설정

```bash
cargo new blog-api
cd blog-api
```

```toml
# Cargo.toml
[package]
name = "blog-api"
version = "0.1.0"
edition = "2021"

[dependencies]
# 웹 프레임워크
axum = { version = "0.8", features = ["macros"] }
tokio = { version = "1", features = ["full"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "compression-full", "trace"] }

# 데이터베이스
sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite", "migrate"] }

# 직렬화
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# 인증
jsonwebtoken = "9"
argon2 = "0.5"

# 유효성 검증
validator = { version = "0.18", features = ["derive"] }

# 에러 처리
thiserror = "2"
anyhow = "1"

# 로깅
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# 유틸리티
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4"] }
dotenvy = "0.15"

[dev-dependencies]
axum-test = "16"
```

---

## 3. 데이터베이스 스키마와 마이그레이션

```sql
-- migrations/001_initial.sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    bio TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    body TEXT NOT NULL,
    published BOOLEAN NOT NULL DEFAULT FALSE,
    author_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    body TEXT NOT NULL,
    post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_posts_author ON posts(author_id);
CREATE INDEX idx_posts_slug ON posts(slug);
CREATE INDEX idx_comments_post ON comments(post_id);
```

---

## 4. 애플리케이션 아키텍처

### 설정

```rust
// src/config.rs
use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub database_url: String,
    pub jwt_secret: String,
    pub server_host: String,
    pub server_port: u16,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        dotenvy::dotenv().ok();

        Ok(Config {
            database_url: env::var("DATABASE_URL")
                .unwrap_or_else(|_| "sqlite:blog.db?mode=rwc".into()),
            jwt_secret: env::var("JWT_SECRET")
                .unwrap_or_else(|_| "development-secret-change-in-production".into()),
            server_host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            server_port: env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(3000),
        })
    }

    pub fn server_addr(&self) -> String {
        format!("{}:{}", self.server_host, self.server_port)
    }
}
```

### 애플리케이션 상태

```rust
// src/main.rs
use sqlx::SqlitePool;

#[derive(Clone)]
pub struct AppState {
    pub db: SqlitePool,
    pub config: Config,
}
```

### 진입점

```rust
// src/main.rs
mod config;
mod db;
mod error;
mod auth;
mod models;
mod routes;
mod middleware;

use crate::config::Config;
use sqlx::sqlite::SqlitePoolOptions;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 로깅 초기화
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("blog_api=debug".parse()?))
        .init();

    let config = Config::from_env()?;

    // 데이터베이스 설정
    let pool = SqlitePoolOptions::new()
        .max_connections(10)
        .connect(&config.database_url)
        .await?;

    sqlx::migrate!("./migrations")
        .run(&pool)
        .await?;

    let state = AppState {
        db: pool,
        config: config.clone(),
    };

    // 라우터 구성
    let app = routes::create_router(state);

    // 서버 시작
    let addr = config.server_addr();
    tracing::info!("{addr}에서 서버 시작");
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("셧다운 신호 수신");
}
```

---

## 5. 데이터베이스 모델

```rust
// src/models/user.rs
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct User {
    pub id: i64,
    pub username: String,
    pub email: String,
    #[serde(skip_serializing)]
    pub password_hash: String,
    pub bio: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct UserProfile {
    pub id: i64,
    pub username: String,
    pub bio: String,
    pub created_at: String,
}

impl User {
    pub async fn find_by_id(pool: &SqlitePool, id: i64) -> sqlx::Result<Option<Self>> {
        sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?")
            .bind(id)
            .fetch_optional(pool)
            .await
    }

    pub async fn find_by_username(pool: &SqlitePool, username: &str) -> sqlx::Result<Option<Self>> {
        sqlx::query_as::<_, User>("SELECT * FROM users WHERE username = ?")
            .bind(username)
            .fetch_optional(pool)
            .await
    }

    pub async fn find_by_email(pool: &SqlitePool, email: &str) -> sqlx::Result<Option<Self>> {
        sqlx::query_as::<_, User>("SELECT * FROM users WHERE email = ?")
            .bind(email)
            .fetch_optional(pool)
            .await
    }

    pub async fn create(
        pool: &SqlitePool,
        username: &str,
        email: &str,
        password_hash: &str,
    ) -> sqlx::Result<Self> {
        sqlx::query_as::<_, User>(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?) RETURNING *"
        )
        .bind(username)
        .bind(email)
        .bind(password_hash)
        .fetch_one(pool)
        .await
    }

    pub fn to_profile(&self) -> UserProfile {
        UserProfile {
            id: self.id,
            username: self.username.clone(),
            bio: self.bio.clone(),
            created_at: self.created_at.clone(),
        }
    }
}
```

```rust
// src/models/post.rs
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct Post {
    pub id: i64,
    pub title: String,
    pub slug: String,
    pub body: String,
    pub published: bool,
    pub author_id: i64,
    pub created_at: String,
    pub updated_at: String,
}

impl Post {
    pub async fn list(
        pool: &SqlitePool,
        page: i64,
        per_page: i64,
    ) -> sqlx::Result<(Vec<Post>, i64)> {
        let offset = (page - 1) * per_page;

        let posts = sqlx::query_as::<_, Post>(
            "SELECT * FROM posts WHERE published = TRUE ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        .bind(per_page)
        .bind(offset)
        .fetch_all(pool)
        .await?;

        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM posts WHERE published = TRUE"
        )
        .fetch_one(pool)
        .await?;

        Ok((posts, count.0))
    }

    pub async fn find_by_id(pool: &SqlitePool, id: i64) -> sqlx::Result<Option<Self>> {
        sqlx::query_as::<_, Post>("SELECT * FROM posts WHERE id = ?")
            .bind(id)
            .fetch_optional(pool)
            .await
    }

    pub async fn create(
        pool: &SqlitePool,
        title: &str,
        slug: &str,
        body: &str,
        author_id: i64,
    ) -> sqlx::Result<Self> {
        sqlx::query_as::<_, Post>(
            "INSERT INTO posts (title, slug, body, author_id, published) \
             VALUES (?, ?, ?, ?, TRUE) RETURNING *"
        )
        .bind(title)
        .bind(slug)
        .bind(body)
        .bind(author_id)
        .fetch_one(pool)
        .await
    }

    pub async fn update(
        pool: &SqlitePool,
        id: i64,
        title: &str,
        body: &str,
    ) -> sqlx::Result<Self> {
        sqlx::query_as::<_, Post>(
            "UPDATE posts SET title = ?, body = ?, updated_at = datetime('now') \
             WHERE id = ? RETURNING *"
        )
        .bind(title)
        .bind(body)
        .bind(id)
        .fetch_one(pool)
        .await
    }

    pub async fn delete(pool: &SqlitePool, id: i64) -> sqlx::Result<()> {
        sqlx::query("DELETE FROM posts WHERE id = ?")
            .bind(id)
            .execute(pool)
            .await?;
        Ok(())
    }
}
```

---

## 6. 에러 처리 레이어

```rust
// src/error.rs
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("찾을 수 없음: {0}")]
    NotFound(String),

    #[error("잘못된 요청: {0}")]
    BadRequest(String),

    #[error("인증 실패")]
    Unauthorized,

    #[error("접근 금지: {0}")]
    Forbidden(String),

    #[error("충돌: {0}")]
    Conflict(String),

    #[error("내부 서버 에러")]
    Internal(#[from] anyhow::Error),

    #[error("데이터베이스 에러")]
    Database(#[from] sqlx::Error),
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error, detail) = match &self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "Not Found", Some(msg.clone())),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "Bad Request", Some(msg.clone())),
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized", None),
            ApiError::Forbidden(msg) => (StatusCode::FORBIDDEN, "Forbidden", Some(msg.clone())),
            ApiError::Conflict(msg) => (StatusCode::CONFLICT, "Conflict", Some(msg.clone())),
            ApiError::Internal(e) => {
                tracing::error!("내부 에러: {e:#}");
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error", None)
            }
            ApiError::Database(e) => {
                tracing::error!("데이터베이스 에러: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error", None)
            }
        };

        let body = ErrorResponse {
            error: error.to_string(),
            detail,
        };

        (status, Json(body)).into_response()
    }
}
```

---

## 7. 인증

```rust
// src/auth/password.rs
use argon2::{
    password_hash::{rand_core::OsRng, SaltString, PasswordHash, PasswordHasher, PasswordVerifier},
    Argon2,
};

pub fn hash_password(password: &str) -> anyhow::Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    let hash = Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| anyhow::anyhow!("비밀번호 해싱 실패: {e}"))?
        .to_string();
    Ok(hash)
}

pub fn verify_password(password: &str, hash: &str) -> anyhow::Result<bool> {
    let parsed_hash = PasswordHash::new(hash)
        .map_err(|e| anyhow::anyhow!("잘못된 비밀번호 해시: {e}"))?;
    Ok(Argon2::default().verify_password(password.as_bytes(), &parsed_hash).is_ok())
}
```

```rust
// src/auth/jwt.rs
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use serde::{Serialize, Deserialize};
use chrono::{Utc, Duration};

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: i64,       // user_id
    pub username: String,
    pub exp: usize,     // 만료 타임스탬프
    pub iat: usize,     // 발급 시간
}

pub fn create_token(user_id: i64, username: &str, secret: &str) -> anyhow::Result<String> {
    let now = Utc::now();
    let claims = Claims {
        sub: user_id,
        username: username.to_string(),
        exp: (now + Duration::hours(24)).timestamp() as usize,
        iat: now.timestamp() as usize,
    };

    let token = encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )?;

    Ok(token)
}

pub fn verify_token(token: &str, secret: &str) -> anyhow::Result<Claims> {
    let data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )?;

    Ok(data.claims)
}
```

```rust
// src/auth/middleware.rs
use axum::{
    extract::{FromRequestParts, State},
    http::{request::Parts, header::AUTHORIZATION},
};
use crate::{AppState, error::ApiError, auth::jwt};

pub struct AuthUser {
    pub user_id: i64,
    pub username: String,
}

impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
    AppState: FromRef<S>,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let app_state = AppState::from_ref(state);

        let header = parts.headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or(ApiError::Unauthorized)?;

        let token = header
            .strip_prefix("Bearer ")
            .ok_or(ApiError::Unauthorized)?;

        let claims = jwt::verify_token(token, &app_state.config.jwt_secret)
            .map_err(|_| ApiError::Unauthorized)?;

        Ok(AuthUser {
            user_id: claims.sub,
            username: claims.username,
        })
    }
}

use axum::extract::FromRef;
```

---

## 8. API 라우트: 사용자

```rust
// src/routes/auth.rs
use axum::{extract::State, Json};
use serde::Deserialize;
use validator::Validate;
use crate::{AppState, error::ApiError, models::user::User, auth};

#[derive(Deserialize, Validate)]
pub struct RegisterRequest {
    #[validate(length(min = 3, max = 32))]
    pub username: String,
    #[validate(email)]
    pub email: String,
    #[validate(length(min = 8))]
    pub password: String,
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(serde::Serialize)]
pub struct AuthResponse {
    pub token: String,
    pub user: crate::models::user::UserProfile,
}

pub async fn register(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<AuthResponse>, ApiError> {
    req.validate().map_err(|e| ApiError::BadRequest(e.to_string()))?;

    // 기존 사용자 확인
    if User::find_by_username(&state.db, &req.username).await?.is_some() {
        return Err(ApiError::Conflict("사용자명이 이미 사용 중입니다".into()));
    }
    if User::find_by_email(&state.db, &req.email).await?.is_some() {
        return Err(ApiError::Conflict("이메일이 이미 등록되어 있습니다".into()));
    }

    let password_hash = auth::password::hash_password(&req.password)
        .map_err(|e| ApiError::Internal(e))?;

    let user = User::create(&state.db, &req.username, &req.email, &password_hash).await?;

    let token = auth::jwt::create_token(user.id, &user.username, &state.config.jwt_secret)
        .map_err(|e| ApiError::Internal(e))?;

    Ok(Json(AuthResponse {
        token,
        user: user.to_profile(),
    }))
}

pub async fn login(
    State(state): State<AppState>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<AuthResponse>, ApiError> {
    let user = User::find_by_username(&state.db, &req.username)
        .await?
        .ok_or(ApiError::Unauthorized)?;

    let valid = auth::password::verify_password(&req.password, &user.password_hash)
        .map_err(|e| ApiError::Internal(e))?;

    if !valid {
        return Err(ApiError::Unauthorized);
    }

    let token = auth::jwt::create_token(user.id, &user.username, &state.config.jwt_secret)
        .map_err(|e| ApiError::Internal(e))?;

    Ok(Json(AuthResponse {
        token,
        user: user.to_profile(),
    }))
}
```

---

## 9. API 라우트: 게시물

```rust
// src/routes/posts.rs
use axum::{extract::{Path, Query, State}, Json};
use serde::{Deserialize, Serialize};
use crate::{AppState, error::ApiError, models::post::Post, auth::middleware::AuthUser};

#[derive(Deserialize)]
pub struct ListParams {
    #[serde(default = "default_page")]
    page: i64,
    #[serde(default = "default_per_page")]
    per_page: i64,
}

fn default_page() -> i64 { 1 }
fn default_per_page() -> i64 { 20 }

#[derive(Serialize)]
pub struct PaginatedPosts {
    data: Vec<Post>,
    page: i64,
    per_page: i64,
    total: i64,
    total_pages: i64,
}

pub async fn list_posts(
    State(state): State<AppState>,
    Query(params): Query<ListParams>,
) -> Result<Json<PaginatedPosts>, ApiError> {
    let per_page = params.per_page.min(100).max(1);
    let page = params.page.max(1);

    let (posts, total) = Post::list(&state.db, page, per_page).await?;

    Ok(Json(PaginatedPosts {
        data: posts,
        page,
        per_page,
        total,
        total_pages: (total + per_page - 1) / per_page,
    }))
}

#[derive(Deserialize)]
pub struct CreatePost {
    pub title: String,
    pub body: String,
}

fn slugify(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

pub async fn create_post(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(req): Json<CreatePost>,
) -> Result<(axum::http::StatusCode, Json<Post>), ApiError> {
    if req.title.trim().is_empty() {
        return Err(ApiError::BadRequest("제목은 비어있을 수 없습니다".into()));
    }

    let slug = format!("{}-{}", slugify(&req.title), uuid::Uuid::new_v4().to_string().split('-').next().unwrap());

    let post = Post::create(&state.db, &req.title, &slug, &req.body, auth.user_id).await?;

    Ok((axum::http::StatusCode::CREATED, Json(post)))
}

pub async fn get_post(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<Post>, ApiError> {
    Post::find_by_id(&state.db, id)
        .await?
        .map(Json)
        .ok_or(ApiError::NotFound(format!("게시물 {id}를 찾을 수 없음")))
}

pub async fn update_post(
    State(state): State<AppState>,
    auth: AuthUser,
    Path(id): Path<i64>,
    Json(req): Json<CreatePost>,
) -> Result<Json<Post>, ApiError> {
    let existing = Post::find_by_id(&state.db, id)
        .await?
        .ok_or(ApiError::NotFound(format!("게시물 {id}를 찾을 수 없음")))?;

    if existing.author_id != auth.user_id {
        return Err(ApiError::Forbidden("자신의 게시물만 수정할 수 있습니다".into()));
    }

    let post = Post::update(&state.db, id, &req.title, &req.body).await?;
    Ok(Json(post))
}

pub async fn delete_post(
    State(state): State<AppState>,
    auth: AuthUser,
    Path(id): Path<i64>,
) -> Result<axum::http::StatusCode, ApiError> {
    let existing = Post::find_by_id(&state.db, id)
        .await?
        .ok_or(ApiError::NotFound(format!("게시물 {id}를 찾을 수 없음")))?;

    if existing.author_id != auth.user_id {
        return Err(ApiError::Forbidden("자신의 게시물만 삭제할 수 있습니다".into()));
    }

    Post::delete(&state.db, id).await?;
    Ok(axum::http::StatusCode::NO_CONTENT)
}
```

---

## 10. API 라우트: 댓글

```rust
// src/routes/comments.rs
use axum::{extract::{Path, State}, Json};
use serde::Deserialize;
use crate::{AppState, error::ApiError, auth::middleware::AuthUser};

#[derive(sqlx::FromRow, serde::Serialize)]
pub struct Comment {
    pub id: i64,
    pub body: String,
    pub post_id: i64,
    pub author_id: i64,
    pub created_at: String,
}

#[derive(Deserialize)]
pub struct CreateComment {
    pub body: String,
}

pub async fn create_comment(
    State(state): State<AppState>,
    auth: AuthUser,
    Path(post_id): Path<i64>,
    Json(req): Json<CreateComment>,
) -> Result<(axum::http::StatusCode, Json<Comment>), ApiError> {
    if req.body.trim().is_empty() {
        return Err(ApiError::BadRequest("댓글 내용은 비어있을 수 없습니다".into()));
    }

    // 게시물 존재 확인
    crate::models::post::Post::find_by_id(&state.db, post_id)
        .await?
        .ok_or(ApiError::NotFound(format!("게시물 {post_id}를 찾을 수 없음")))?;

    let comment = sqlx::query_as::<_, Comment>(
        "INSERT INTO comments (body, post_id, author_id) VALUES (?, ?, ?) RETURNING *"
    )
    .bind(&req.body)
    .bind(post_id)
    .bind(auth.user_id)
    .fetch_one(&state.db)
    .await?;

    Ok((axum::http::StatusCode::CREATED, Json(comment)))
}

pub async fn list_comments(
    State(state): State<AppState>,
    Path(post_id): Path<i64>,
) -> Result<Json<Vec<Comment>>, ApiError> {
    let comments = sqlx::query_as::<_, Comment>(
        "SELECT * FROM comments WHERE post_id = ? ORDER BY created_at ASC"
    )
    .bind(post_id)
    .fetch_all(&state.db)
    .await?;

    Ok(Json(comments))
}
```

---

## 11. 미들웨어 스택

```rust
// src/routes/mod.rs
use axum::{
    routing::{get, post, put, delete},
    Router,
    middleware as axum_middleware,
};
use tower_http::{
    cors::CorsLayer,
    compression::CompressionLayer,
    trace::TraceLayer,
};
use crate::AppState;

pub mod auth;
pub mod posts;
pub mod comments;

pub fn create_router(state: AppState) -> Router {
    let api = Router::new()
        // 인증이 필요 없는 라우트
        .route("/auth/register", post(auth::register))
        .route("/auth/login", post(auth::login))
        // 공개 게시물 라우트
        .route("/posts", get(posts::list_posts))
        .route("/posts/{id}", get(posts::get_post))
        .route("/posts/{post_id}/comments", get(comments::list_comments))
        // 보호된 라우트 (추출기를 통해 인증 필요)
        .route("/posts", post(posts::create_post))
        .route("/posts/{id}", put(posts::update_post).delete(posts::delete_post))
        .route("/posts/{post_id}/comments", post(comments::create_comment));

    Router::new()
        .route("/health", get(|| async { "OK" }))
        .nest("/api", api)
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
```

---

## 12. 테스팅

```rust
// tests/posts_test.rs
use axum_test::TestServer;
use serde_json::json;

async fn setup() -> TestServer {
    let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();

    let config = Config {
        database_url: "sqlite::memory:".into(),
        jwt_secret: "test-secret".into(),
        server_host: "127.0.0.1".into(),
        server_port: 0,
    };

    let state = AppState { db: pool, config };
    let app = routes::create_router(state);
    TestServer::new(app).unwrap()
}

#[tokio::test]
async fn test_register_and_login() {
    let server = setup().await;

    // 회원가입
    let resp = server.post("/api/auth/register")
        .json(&json!({
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123"
        }))
        .await;

    resp.assert_status_ok();
    let body: serde_json::Value = resp.json();
    assert!(body["token"].is_string());

    // 로그인
    let resp = server.post("/api/auth/login")
        .json(&json!({
            "username": "testuser",
            "password": "password123"
        }))
        .await;

    resp.assert_status_ok();
}

#[tokio::test]
async fn test_create_and_list_posts() {
    let server = setup().await;

    // 회원가입 후 토큰 획득
    let resp = server.post("/api/auth/register")
        .json(&json!({
            "username": "author",
            "email": "author@example.com",
            "password": "password123"
        }))
        .await;
    let token: String = resp.json::<serde_json::Value>()["token"]
        .as_str().unwrap().to_string();

    // 게시물 작성
    let resp = server.post("/api/posts")
        .add_header("Authorization".parse().unwrap(), format!("Bearer {token}").parse().unwrap())
        .json(&json!({
            "title": "첫 번째 게시물",
            "body": "이것은 내용입니다"
        }))
        .await;

    resp.assert_status(axum::http::StatusCode::CREATED);

    // 게시물 목록 조회
    let resp = server.get("/api/posts").await;
    resp.assert_status_ok();
    let body: serde_json::Value = resp.json();
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert_eq!(body["total"], 1);
}

use crate::{config::Config, AppState, routes};
```

---

## 13. 배포

### Docker

```dockerfile
# 빌드 단계
FROM rust:1.82 AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

# 런타임 단계
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/blog-api /usr/local/bin/
COPY --from=builder /app/migrations /app/migrations

ENV DATABASE_URL=sqlite:/data/blog.db?mode=rwc
ENV PORT=3000
EXPOSE 3000
VOLUME ["/data"]

CMD ["blog-api"]
```

### 환경 변수

```bash
# .env
DATABASE_URL=sqlite:blog.db?mode=rwc
JWT_SECRET=이것을-랜덤-문자열로-변경하세요
HOST=0.0.0.0
PORT=3000
RUST_LOG=blog_api=info,tower_http=debug
```

---

## 14. 확장 아이디어

**전문 검색(Full-text search)** — 제목과 본문으로 게시물을 검색하기 위한 SQLite FTS5 추가.

**속도 제한** — IP별 토큰 버킷을 이용한 Tower 속도 제한 레이어 추가.

**이미지 업로드** — 멀티파트 폼 데이터를 받아 이미지를 디스크에 저장하고 URL을 반환하는 `/api/upload` 엔드포인트 추가.

**이메일 인증** — 이메일로 전송된 토큰을 사용하여 회원가입 후 이메일 인증 요구.

**RSS 피드** — 게시된 게시물로 RSS 피드를 생성하는 `/feed.xml` 엔드포인트 추가.

**OpenAPI 스펙** — `utoipa`를 사용하여 OpenAPI 명세와 Swagger UI 자동 생성.

**WebSocket 알림** — 새 게시물이나 댓글이 생성될 때 연결된 클라이언트에게 알림.

**캐싱** — 자주 접근하는 게시물에 인메모리 캐시(예: `moka`) 추가.

---

## 참고 자료

- [Axum documentation](https://docs.rs/axum/latest/axum/)
- [SQLx documentation](https://docs.rs/sqlx/latest/sqlx/)
- [jsonwebtoken documentation](https://docs.rs/jsonwebtoken/latest/jsonwebtoken/)
- [argon2 documentation](https://docs.rs/argon2/latest/argon2/)
- [tower-http documentation](https://docs.rs/tower-http/latest/tower_http/)
- [Zero To Production In Rust (book)](https://www.zero2prod.com/)

---

**이전**: [성능과 프로파일링](./13_Performance_Profiling.md)
