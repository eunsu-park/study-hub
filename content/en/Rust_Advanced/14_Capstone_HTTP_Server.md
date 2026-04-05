# 30. Capstone: HTTP Server

**Previous**: [Performance and Profiling](./13_Performance_Profiling.md)

## Learning Objectives

After completing this project, you will be able to:

1. Architect a complete HTTP server with routing, middleware, and error handling
2. Implement JSON API endpoints with validation and pagination
3. Integrate a database with connection pooling and migrations
4. Add authentication with JWT tokens and password hashing
5. Prepare the application for production deployment

---

This is a capstone project. Instead of learning individual concepts, you will build a complete **REST API server** from scratch. The project ties together nearly every concept from the course: ownership, traits, async, error handling, macros, and networking. You will build a production-quality blog API with users, posts, comments, authentication, and database storage.

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Project Setup](#2-project-setup)
3. [Database Schema and Migrations](#3-database-schema-and-migrations)
4. [Application Architecture](#4-application-architecture)
5. [Database Models](#5-database-models)
6. [Error Handling Layer](#6-error-handling-layer)
7. [Authentication](#7-authentication)
8. [API Routes: Users](#8-api-routes-users)
9. [API Routes: Posts](#9-api-routes-posts)
10. [API Routes: Comments](#10-api-routes-comments)
11. [Middleware Stack](#11-middleware-stack)
12. [Testing](#12-testing)
13. [Deployment](#13-deployment)
14. [Extension Ideas](#14-extension-ideas)

---

## 1. Project Overview

Here is what the finished API looks like:

```
POST   /api/auth/register     — Create account
POST   /api/auth/login        — Get JWT token
GET    /api/users/:id         — Get user profile
GET    /api/posts             — List posts (paginated)
POST   /api/posts             — Create post (auth required)
GET    /api/posts/:id         — Get post with comments
PUT    /api/posts/:id         — Update post (owner only)
DELETE /api/posts/:id         — Delete post (owner only)
POST   /api/posts/:id/comments — Add comment (auth required)
GET    /health                — Health check
```

### Technology Stack

| Component | Choice | Why |
|-----------|--------|-----|
| HTTP | Axum | Ergonomic, Tower-based, fastest Rust web framework |
| Database | SQLite + SQLx | Zero-config, async, compile-time query checking |
| Auth | JWT (jsonwebtoken) | Stateless, standard |
| Password | Argon2 (argon2) | OWASP recommended |
| Validation | validator | Derive-based struct validation |
| Error handling | thiserror + anyhow | Typed API errors, contextual internals |
| Logging | tracing | Structured, async-aware |
| Testing | axum-test | In-process HTTP testing |

### Architecture

```
blog-api/
├── Cargo.toml
├── migrations/
│   └── 001_initial.sql
├── src/
│   ├── main.rs           ← Entry point, server startup
│   ├── config.rs          ← Configuration from environment
│   ├── db.rs              ← Database pool setup
│   ├── error.rs           ← Error types and conversions
│   ├── auth/
│   │   ├── mod.rs         ← Auth module root
│   │   ├── jwt.rs         ← JWT token creation/validation
│   │   ├── password.rs    ← Password hashing
│   │   └── middleware.rs  ← Auth extraction middleware
│   ├── models/
│   │   ├── mod.rs
│   │   ├── user.rs        ← User queries
│   │   ├── post.rs        ← Post queries
│   │   └── comment.rs     ← Comment queries
│   ├── routes/
│   │   ├── mod.rs         ← Router assembly
│   │   ├── auth.rs        ← /api/auth/* handlers
│   │   ├── users.rs       ← /api/users/* handlers
│   │   ├── posts.rs       ← /api/posts/* handlers
│   │   └── comments.rs    ← /api/posts/*/comments handlers
│   └── middleware.rs      ← Custom middleware (timing, logging)
└── tests/
    ├── auth_test.rs
    ├── posts_test.rs
    └── common/mod.rs      ← Test helpers
```

---

## 2. Project Setup

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
# Web framework
axum = { version = "0.8", features = ["macros"] }
tokio = { version = "1", features = ["full"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "compression-full", "trace"] }

# Database
sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite", "migrate"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Authentication
jsonwebtoken = "9"
argon2 = "0.5"

# Validation
validator = { version = "0.18", features = ["derive"] }

# Error handling
thiserror = "2"
anyhow = "1"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Utilities
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4"] }
dotenvy = "0.15"

[dev-dependencies]
axum-test = "16"
```

---

## 3. Database Schema and Migrations

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

## 4. Application Architecture

### Configuration

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

### Application State

```rust
// src/main.rs
use sqlx::SqlitePool;

#[derive(Clone)]
pub struct AppState {
    pub db: SqlitePool,
    pub config: Config,
}
```

### Entry Point

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
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("blog_api=debug".parse()?))
        .init();

    let config = Config::from_env()?;

    // Database setup
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

    // Build router
    let app = routes::create_router(state);

    // Start server
    let addr = config.server_addr();
    tracing::info!("Starting server on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("Shutdown signal received");
}
```

---

## 5. Database Models

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

## 6. Error Handling Layer

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
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Unauthorized")]
    Unauthorized,

    #[error("Forbidden: {0}")]
    Forbidden(String),

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Internal server error")]
    Internal(#[from] anyhow::Error),

    #[error("Database error")]
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
                tracing::error!("Internal error: {e:#}");
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error", None)
            }
            ApiError::Database(e) => {
                tracing::error!("Database error: {e}");
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

## 7. Authentication

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
        .map_err(|e| anyhow::anyhow!("Password hashing failed: {e}"))?
        .to_string();
    Ok(hash)
}

pub fn verify_password(password: &str, hash: &str) -> anyhow::Result<bool> {
    let parsed_hash = PasswordHash::new(hash)
        .map_err(|e| anyhow::anyhow!("Invalid password hash: {e}"))?;
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
    pub exp: usize,     // expiry timestamp
    pub iat: usize,     // issued at
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

## 8. API Routes: Users

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

    // Check for existing user
    if User::find_by_username(&state.db, &req.username).await?.is_some() {
        return Err(ApiError::Conflict("Username already taken".into()));
    }
    if User::find_by_email(&state.db, &req.email).await?.is_some() {
        return Err(ApiError::Conflict("Email already registered".into()));
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

## 9. API Routes: Posts

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
        return Err(ApiError::BadRequest("Title cannot be empty".into()));
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
        .ok_or(ApiError::NotFound(format!("Post {id} not found")))
}

pub async fn update_post(
    State(state): State<AppState>,
    auth: AuthUser,
    Path(id): Path<i64>,
    Json(req): Json<CreatePost>,
) -> Result<Json<Post>, ApiError> {
    let existing = Post::find_by_id(&state.db, id)
        .await?
        .ok_or(ApiError::NotFound(format!("Post {id} not found")))?;

    if existing.author_id != auth.user_id {
        return Err(ApiError::Forbidden("You can only edit your own posts".into()));
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
        .ok_or(ApiError::NotFound(format!("Post {id} not found")))?;

    if existing.author_id != auth.user_id {
        return Err(ApiError::Forbidden("You can only delete your own posts".into()));
    }

    Post::delete(&state.db, id).await?;
    Ok(axum::http::StatusCode::NO_CONTENT)
}
```

---

## 10. API Routes: Comments

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
        return Err(ApiError::BadRequest("Comment body cannot be empty".into()));
    }

    // Verify post exists
    crate::models::post::Post::find_by_id(&state.db, post_id)
        .await?
        .ok_or(ApiError::NotFound(format!("Post {post_id} not found")))?;

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

## 11. Middleware Stack

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
        // Auth routes (no authentication required)
        .route("/auth/register", post(auth::register))
        .route("/auth/login", post(auth::login))
        // Public post routes
        .route("/posts", get(posts::list_posts))
        .route("/posts/{id}", get(posts::get_post))
        .route("/posts/{post_id}/comments", get(comments::list_comments))
        // Protected routes (authentication required via extractor)
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

## 12. Testing

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

    // Register
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

    // Login
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

    // Register and get token
    let resp = server.post("/api/auth/register")
        .json(&json!({
            "username": "author",
            "email": "author@example.com",
            "password": "password123"
        }))
        .await;
    let token: String = resp.json::<serde_json::Value>()["token"]
        .as_str().unwrap().to_string();

    // Create post
    let resp = server.post("/api/posts")
        .add_header("Authorization".parse().unwrap(), format!("Bearer {token}").parse().unwrap())
        .json(&json!({
            "title": "My First Post",
            "body": "This is the content"
        }))
        .await;

    resp.assert_status(axum::http::StatusCode::CREATED);

    // List posts
    let resp = server.get("/api/posts").await;
    resp.assert_status_ok();
    let body: serde_json::Value = resp.json();
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert_eq!(body["total"], 1);
}

use crate::{config::Config, AppState, routes};
```

---

## 13. Deployment

### Docker

```dockerfile
# Build stage
FROM rust:1.82 AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

# Runtime stage
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

### Environment Variables

```bash
# .env
DATABASE_URL=sqlite:blog.db?mode=rwc
JWT_SECRET=change-this-to-a-random-string
HOST=0.0.0.0
PORT=3000
RUST_LOG=blog_api=info,tower_http=debug
```

---

## 14. Extension Ideas

**Full-text search** — Add SQLite FTS5 for searching posts by title and body.

**Rate limiting** — Add a Tower rate-limiting layer with per-IP token buckets.

**Image upload** — Add a `/api/upload` endpoint that accepts multipart form data, stores images on disk, and returns URLs.

**Email verification** — Require email verification after registration using a token sent via email.

**RSS feed** — Add a `/feed.xml` endpoint that generates an RSS feed from published posts.

**OpenAPI spec** — Use `utoipa` to auto-generate an OpenAPI specification and Swagger UI.

**WebSocket notifications** — Notify connected clients when new posts or comments are created.

**Caching** — Add an in-memory cache (e.g., `moka`) for frequently accessed posts.

---

## References

- [Axum documentation](https://docs.rs/axum/latest/axum/)
- [SQLx documentation](https://docs.rs/sqlx/latest/sqlx/)
- [jsonwebtoken documentation](https://docs.rs/jsonwebtoken/latest/jsonwebtoken/)
- [argon2 documentation](https://docs.rs/argon2/latest/argon2/)
- [tower-http documentation](https://docs.rs/tower-http/latest/tower_http/)
- [Zero To Production In Rust (book)](https://www.zero2prod.com/)

---

**Previous**: [Performance and Profiling](./13_Performance_Profiling.md)
