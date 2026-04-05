# 22. Capstone: Microservice

**Previous**: [Cloud Native Patterns](./10_Cloud_Native_Patterns.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design and build a complete microservice from scratch
2. Integrate all patterns from previous lessons into a cohesive application
3. Implement CRUD, authentication, middleware, and observability
4. Deploy with Docker and implement CI/CD
5. Write comprehensive tests including integration tests

---

This capstone lesson brings together everything from the course. We will build a complete URL shortener microservice with a REST API, PostgreSQL persistence, Redis caching, authentication, rate limiting, and full observability.

## Table of Contents
1. [Architecture Design](#1-architecture-design)
2. [Project Structure](#2-project-structure)
3. [Core Domain](#3-core-domain)
4. [HTTP Layer](#4-http-layer)
5. [Persistence Layer](#5-persistence-layer)
6. [Testing](#6-testing)
7. [Summary](#7-summary)

---

## 1. Architecture Design

### 1.1 System Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│   Client     │────▶│  HTTP Server │────▶│ Service  │
│ (browser/API)│◀────│  (handlers)  │◀────│ (logic)  │
└─────────────┘     └──────────────┘     └──────────┘
                          │                     │
                    ┌─────┴─────┐         ┌─────┴─────┐
                    │Middleware │         │Repository │
                    │- logging  │         │- postgres │
                    │- auth     │         │- redis    │
                    │- rate limit│         └───────────┘
                    │- recovery │
                    └───────────┘
```

### 1.2 API Design

```
POST   /api/v1/urls          — Create short URL
GET    /api/v1/urls           — List user's URLs
GET    /api/v1/urls/{code}    — Get URL details
DELETE /api/v1/urls/{code}    — Delete short URL
GET    /api/v1/urls/{code}/stats — Get click statistics

GET    /{code}                — Redirect to original URL

POST   /api/v1/auth/register  — Register user
POST   /api/v1/auth/login     — Login, get JWT
GET    /healthz               — Liveness probe
GET    /readyz                 — Readiness probe
```

### 1.3 Data Models

```go
type URL struct {
    ID          string    `json:"id" db:"id"`
    Code        string    `json:"code" db:"code"`
    OriginalURL string    `json:"original_url" db:"original_url"`
    UserID      string    `json:"user_id" db:"user_id"`
    Clicks      int64     `json:"clicks" db:"clicks"`
    ExpiresAt   *time.Time `json:"expires_at,omitempty" db:"expires_at"`
    CreatedAt   time.Time `json:"created_at" db:"created_at"`
}

type User struct {
    ID           string    `json:"id" db:"id"`
    Email        string    `json:"email" db:"email"`
    PasswordHash string    `json:"-" db:"password_hash"`
    CreatedAt    time.Time `json:"created_at" db:"created_at"`
}

type ClickEvent struct {
    ID        string    `json:"id" db:"id"`
    URLID     string    `json:"url_id" db:"url_id"`
    IP        string    `json:"ip" db:"ip"`
    UserAgent string    `json:"user_agent" db:"user_agent"`
    Referer   string    `json:"referer" db:"referer"`
    CreatedAt time.Time `json:"created_at" db:"created_at"`
}
```

---

## 2. Project Structure

### 2.1 Directory Layout

```
urlshort/
├── cmd/
│   └── server/
│       └── main.go              # Entry point
├── internal/
│   ├── config/
│   │   └── config.go            # Configuration
│   ├── domain/
│   │   ├── url.go               # URL entity and repository interface
│   │   └── user.go              # User entity and repository interface
│   ├── handler/
│   │   ├── url_handler.go       # URL HTTP handlers
│   │   ├── auth_handler.go      # Auth HTTP handlers
│   │   └── health_handler.go    # Health check handlers
│   ├── middleware/
│   │   ├── auth.go              # JWT authentication
│   │   ├── logging.go           # Request logging
│   │   ├── ratelimit.go         # Rate limiting
│   │   └── recovery.go          # Panic recovery
│   ├── repository/
│   │   ├── postgres/
│   │   │   ├── url_repo.go      # PostgreSQL URL repository
│   │   │   └── user_repo.go     # PostgreSQL user repository
│   │   └── redis/
│   │       └── cache.go         # Redis cache
│   ├── service/
│   │   ├── url_service.go       # URL business logic
│   │   └── auth_service.go      # Auth business logic
│   └── server/
│       └── server.go            # HTTP server setup
├── migrations/
│   ├── 001_create_users.sql
│   ├── 002_create_urls.sql
│   └── 003_create_clicks.sql
├── tests/
│   └── integration/
│       └── api_test.go          # Integration tests
├── Dockerfile
├── docker-compose.yml
├── go.mod
├── go.sum
└── Makefile
```

### 2.2 Entry Point

```go
// cmd/server/main.go
package main

import (
    "context"
    "log/slog"
    "os"
    "os/signal"
    "syscall"

    "github.com/user/urlshort/internal/config"
    "github.com/user/urlshort/internal/server"
)

func main() {
    cfg, err := config.Load()
    if err != nil {
        slog.Error("failed to load config", "err", err)
        os.Exit(1)
    }

    logger := setupLogger(cfg.Log)
    slog.SetDefault(logger)

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    srv, err := server.New(ctx, cfg, logger)
    if err != nil {
        slog.Error("failed to create server", "err", err)
        os.Exit(1)
    }

    // Start server
    go func() {
        if err := srv.Start(); err != nil {
            slog.Error("server error", "err", err)
            cancel()
        }
    }()

    // Wait for signal
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

    select {
    case sig := <-quit:
        slog.Info("shutdown signal received", "signal", sig)
    case <-ctx.Done():
    }

    srv.Shutdown()
}
```

---

## 3. Core Domain

### 3.1 URL Service

```go
// internal/service/url_service.go
package service

import (
    "context"
    "crypto/rand"
    "encoding/base62"
    "fmt"
    "time"
)

type URLService struct {
    repo   URLRepository
    cache  Cache
    logger *slog.Logger
}

type URLRepository interface {
    Create(ctx context.Context, url *URL) error
    GetByCode(ctx context.Context, code string) (*URL, error)
    ListByUser(ctx context.Context, userID string, opts ListOptions) ([]URL, int, error)
    Delete(ctx context.Context, code string, userID string) error
    IncrementClicks(ctx context.Context, code string) error
    RecordClick(ctx context.Context, event *ClickEvent) error
}

type Cache interface {
    Get(ctx context.Context, key string) (string, error)
    Set(ctx context.Context, key string, value string, ttl time.Duration) error
    Delete(ctx context.Context, key string) error
}

func NewURLService(repo URLRepository, cache Cache, logger *slog.Logger) *URLService {
    return &URLService{repo: repo, cache: cache, logger: logger}
}

func (s *URLService) Shorten(ctx context.Context, originalURL, userID string, expiresIn *time.Duration) (*URL, error) {
    code := generateCode(7)

    url := &URL{
        ID:          generateID(),
        Code:        code,
        OriginalURL: originalURL,
        UserID:      userID,
        CreatedAt:   time.Now(),
    }
    if expiresIn != nil {
        exp := time.Now().Add(*expiresIn)
        url.ExpiresAt = &exp
    }

    if err := s.repo.Create(ctx, url); err != nil {
        return nil, fmt.Errorf("create url: %w", err)
    }

    // Cache for fast redirects
    s.cache.Set(ctx, "url:"+code, originalURL, 24*time.Hour)

    s.logger.Info("url shortened",
        "code", code,
        "original", originalURL,
        "user_id", userID,
    )

    return url, nil
}

func (s *URLService) Resolve(ctx context.Context, code string, event *ClickEvent) (string, error) {
    // Try cache first
    if cached, err := s.cache.Get(ctx, "url:"+code); err == nil {
        go s.recordClick(context.Background(), code, event)
        return cached, nil
    }

    // Fall back to database
    url, err := s.repo.GetByCode(ctx, code)
    if err != nil {
        return "", fmt.Errorf("resolve %s: %w", code, err)
    }

    // Check expiration
    if url.ExpiresAt != nil && url.ExpiresAt.Before(time.Now()) {
        return "", ErrURLExpired
    }

    // Update cache
    s.cache.Set(ctx, "url:"+code, url.OriginalURL, 24*time.Hour)

    go s.recordClick(context.Background(), code, event)

    return url.OriginalURL, nil
}

func (s *URLService) recordClick(ctx context.Context, code string, event *ClickEvent) {
    s.repo.IncrementClicks(ctx, code)
    if event != nil {
        s.repo.RecordClick(ctx, event)
    }
}

func generateCode(length int) string {
    b := make([]byte, length)
    rand.Read(b)
    return base64.RawURLEncoding.EncodeToString(b)[:length]
}
```

---

## 4. HTTP Layer

### 4.1 URL Handler

```go
// internal/handler/url_handler.go
package handler

type URLHandler struct {
    service *service.URLService
    logger  *slog.Logger
}

func NewURLHandler(svc *service.URLService, logger *slog.Logger) *URLHandler {
    return &URLHandler{service: svc, logger: logger}
}

func (h *URLHandler) Create(w http.ResponseWriter, r *http.Request) {
    var input struct {
        URL       string  `json:"url"`
        ExpiresIn *string `json:"expires_in,omitempty"` // "24h", "7d"
    }

    if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
        writeError(w, http.StatusBadRequest, "invalid JSON")
        return
    }

    if input.URL == "" {
        writeError(w, http.StatusBadRequest, "url is required")
        return
    }

    userID := middleware.UserIDFromContext(r.Context())

    var expiresIn *time.Duration
    if input.ExpiresIn != nil {
        d, err := time.ParseDuration(*input.ExpiresIn)
        if err != nil {
            writeError(w, http.StatusBadRequest, "invalid expires_in format")
            return
        }
        expiresIn = &d
    }

    url, err := h.service.Shorten(r.Context(), input.URL, userID, expiresIn)
    if err != nil {
        h.logger.Error("shorten failed", "err", err)
        writeError(w, http.StatusInternalServerError, "failed to shorten URL")
        return
    }

    writeJSON(w, http.StatusCreated, url)
}

func (h *URLHandler) Redirect(w http.ResponseWriter, r *http.Request) {
    code := r.PathValue("code")

    event := &ClickEvent{
        IP:        r.RemoteAddr,
        UserAgent: r.UserAgent(),
        Referer:   r.Referer(),
    }

    target, err := h.service.Resolve(r.Context(), code, event)
    if err != nil {
        if errors.Is(err, ErrNotFound) || errors.Is(err, ErrURLExpired) {
            http.NotFound(w, r)
            return
        }
        writeError(w, http.StatusInternalServerError, "redirect failed")
        return
    }

    http.Redirect(w, r, target, http.StatusMovedPermanently)
}
```

### 4.2 Router Setup

```go
// internal/server/server.go
func (s *Server) setupRoutes() {
    mux := http.NewServeMux()

    // Middleware chain
    handler := chain(mux,
        s.middleware.Recovery,
        s.middleware.RequestID,
        s.middleware.Logging,
        s.middleware.CORS,
        s.middleware.RateLimit,
    )

    // Health
    mux.HandleFunc("GET /healthz", s.health.Liveness)
    mux.HandleFunc("GET /readyz", s.health.Readiness)

    // Public
    mux.HandleFunc("GET /{code}", s.urlHandler.Redirect)

    // Auth
    mux.HandleFunc("POST /api/v1/auth/register", s.authHandler.Register)
    mux.HandleFunc("POST /api/v1/auth/login", s.authHandler.Login)

    // Protected
    mux.Handle("POST /api/v1/urls", s.middleware.Auth(http.HandlerFunc(s.urlHandler.Create)))
    mux.Handle("GET /api/v1/urls", s.middleware.Auth(http.HandlerFunc(s.urlHandler.List)))
    mux.Handle("GET /api/v1/urls/{code}", s.middleware.Auth(http.HandlerFunc(s.urlHandler.Get)))
    mux.Handle("DELETE /api/v1/urls/{code}", s.middleware.Auth(http.HandlerFunc(s.urlHandler.Delete)))

    s.httpServer.Handler = handler
}
```

---

## 5. Persistence Layer

### 5.1 PostgreSQL Repository

```go
// internal/repository/postgres/url_repo.go
type URLRepo struct {
    db *sql.DB
}

func NewURLRepo(db *sql.DB) *URLRepo {
    return &URLRepo{db: db}
}

func (r *URLRepo) Create(ctx context.Context, url *URL) error {
    _, err := r.db.ExecContext(ctx,
        `INSERT INTO urls (id, code, original_url, user_id, expires_at, created_at)
         VALUES ($1, $2, $3, $4, $5, $6)`,
        url.ID, url.Code, url.OriginalURL, url.UserID, url.ExpiresAt, url.CreatedAt,
    )
    return err
}

func (r *URLRepo) GetByCode(ctx context.Context, code string) (*URL, error) {
    url := &URL{}
    err := r.db.QueryRowContext(ctx,
        `SELECT id, code, original_url, user_id, clicks, expires_at, created_at
         FROM urls WHERE code = $1`, code,
    ).Scan(&url.ID, &url.Code, &url.OriginalURL, &url.UserID,
        &url.Clicks, &url.ExpiresAt, &url.CreatedAt)

    if errors.Is(err, sql.ErrNoRows) {
        return nil, ErrNotFound
    }
    return url, err
}

func (r *URLRepo) IncrementClicks(ctx context.Context, code string) error {
    _, err := r.db.ExecContext(ctx,
        `UPDATE urls SET clicks = clicks + 1 WHERE code = $1`, code)
    return err
}
```

### 5.2 Database Migrations

```sql
-- migrations/001_create_users.sql
CREATE TABLE users (
    id            VARCHAR(36) PRIMARY KEY,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

-- migrations/002_create_urls.sql
CREATE TABLE urls (
    id           VARCHAR(36) PRIMARY KEY,
    code         VARCHAR(20) UNIQUE NOT NULL,
    original_url TEXT NOT NULL,
    user_id      VARCHAR(36) REFERENCES users(id),
    clicks       BIGINT NOT NULL DEFAULT 0,
    expires_at   TIMESTAMP,
    created_at   TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_urls_code ON urls(code);
CREATE INDEX idx_urls_user_id ON urls(user_id);

-- migrations/003_create_clicks.sql
CREATE TABLE click_events (
    id         VARCHAR(36) PRIMARY KEY,
    url_id     VARCHAR(36) REFERENCES urls(id) ON DELETE CASCADE,
    ip         VARCHAR(45),
    user_agent TEXT,
    referer    TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_clicks_url_id ON click_events(url_id);
```

---

## 6. Testing

### 6.1 Unit Tests

```go
func TestURLService_Shorten(t *testing.T) {
    repo := &MockURLRepo{}
    cache := &MockCache{}
    logger := slog.New(slog.NewTextHandler(io.Discard, nil))

    svc := NewURLService(repo, cache, logger)

    url, err := svc.Shorten(context.Background(), "https://example.com", "user-1", nil)
    if err != nil {
        t.Fatal(err)
    }

    if url.OriginalURL != "https://example.com" {
        t.Errorf("got %s, want https://example.com", url.OriginalURL)
    }
    if url.Code == "" {
        t.Error("expected non-empty code")
    }
    if len(repo.Created) != 1 {
        t.Errorf("expected 1 create call, got %d", len(repo.Created))
    }
}
```

### 6.2 Integration Tests

```go
//go:build integration

func TestAPI_CreateAndRedirect(t *testing.T) {
    app := setupTestApp(t)
    defer app.Cleanup()

    // Register
    registerResp := app.POST("/api/v1/auth/register", map[string]string{
        "email":    "test@example.com",
        "password": "secret123",
    })
    assertStatus(t, registerResp, 201)

    // Login
    loginResp := app.POST("/api/v1/auth/login", map[string]string{
        "email":    "test@example.com",
        "password": "secret123",
    })
    assertStatus(t, loginResp, 200)
    token := extractToken(t, loginResp)

    // Create short URL
    createResp := app.POSTAuth("/api/v1/urls", token, map[string]string{
        "url": "https://go.dev",
    })
    assertStatus(t, createResp, 201)
    code := extractCode(t, createResp)

    // Redirect
    redirectResp := app.GETNoFollow("/" + code)
    assertStatus(t, redirectResp, 301)
    assertHeader(t, redirectResp, "Location", "https://go.dev")
}
```

### 6.3 Makefile

```makefile
.PHONY: build test lint docker run migrate

build:
	CGO_ENABLED=0 go build -ldflags="-s -w \
		-X main.version=$(shell git describe --tags --always) \
		-X main.commit=$(shell git rev-parse --short HEAD)" \
		-o bin/server ./cmd/server

test:
	go test -race -cover ./...

test-integration:
	go test -race -tags integration ./tests/...

lint:
	golangci-lint run ./...

docker:
	docker build -t urlshort .

run:
	docker-compose up -d

migrate:
	go run ./cmd/migrate up

clean:
	rm -rf bin/ dist/
```

---

## 7. Summary

### Key Takeaways

1. **Clean architecture** — separate handlers (HTTP), services (business logic), and repositories (persistence).
2. **Interface boundaries** — define interfaces at the consumer for testability and flexibility.
3. **Middleware chain** — logging, auth, rate limiting, recovery in composable layers.
4. **Health checks** — mandatory for cloud deployment. Readiness controls traffic, liveness controls restarts.
5. **Graceful shutdown** — stop accepting traffic, drain requests, close resources.
6. **Comprehensive testing** — unit tests with mocks, integration tests with real database.
7. **Docker and CI/CD** — multi-stage builds, automated testing, release automation.

### Course Summary

This course covered Go from fundamentals to production microservices:

| Phase | Lessons | Key Skills |
|-------|---------|------------|
| Foundations | 01-06 | Types, functions, interfaces, errors, packages |
| Concurrency | 07-09 | Goroutines, channels, patterns, context |
| Tooling | 10-11 | Testing, standard library |
| Web | 12-14 | HTTP, REST, databases |
| Advanced | 15-18 | CLI, generics, reflection, profiling |
| Production | 19-22 | Docker, networking, cloud native, capstone |

You now have the knowledge to build production-ready Go applications. The next step is practice — build projects, read Go source code, and contribute to open-source Go projects.

---

## Exercises

### Exercise 1: Complete the Microservice
Implement the full URL shortener with all components: auth, CRUD, caching, click analytics, and tests.

### Exercise 2: Add Features
Extend with: custom slugs, QR code generation, bulk import, expiration notifications, and an admin dashboard.

### Exercise 3: Performance Testing
Load test with `hey` or `wrk`. Profile, identify bottlenecks, and optimize. Target: 10,000 redirects/second.

### Exercise 4: Deploy to Cloud
Deploy to a cloud provider with: Docker, managed database, CDN, TLS, monitoring, and alerting.
