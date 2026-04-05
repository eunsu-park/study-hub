# 22. 캡스톤: 마이크로서비스

**이전**: [클라우드 네이티브 패턴](./10_Cloud_Native_Patterns.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 완전한 마이크로서비스를 처음부터 설계하고 구축한다
2. 이전 레슨의 모든 패턴을 하나의 응집된 애플리케이션에 통합한다
3. CRUD, 인증, 미들웨어, 관측성을 구현한다
4. Docker로 배포하고 CI/CD를 구현한다
5. 통합 테스트를 포함한 포괄적인 테스트를 작성한다

---

이 캡스톤 레슨은 과정의 모든 내용을 종합한다. REST API, PostgreSQL 영속성, Redis 캐싱, 인증, 요청 제한, 완전한 관측성을 갖춘 URL 단축 마이크로서비스를 구축한다.

## 목차
1. [아키텍처 설계](#1-아키텍처-설계)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [핵심 도메인](#3-핵심-도메인)
4. [HTTP 계층](#4-http-계층)
5. [영속성 계층](#5-영속성-계층)
6. [테스트](#6-테스트)
7. [요약](#7-요약)

---

## 1. 아키텍처 설계

### 1.1 시스템 개요

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

### 1.2 API 설계

```
POST   /api/v1/urls          — 단축 URL 생성
GET    /api/v1/urls           — 사용자의 URL 목록
GET    /api/v1/urls/{code}    — URL 상세 정보
DELETE /api/v1/urls/{code}    — 단축 URL 삭제
GET    /api/v1/urls/{code}/stats — 클릭 통계

GET    /{code}                — 원본 URL로 리다이렉트

POST   /api/v1/auth/register  — 사용자 등록
POST   /api/v1/auth/login     — 로그인, JWT 발급
GET    /healthz               — 라이브니스 프로브
GET    /readyz                 — 레디니스 프로브
```

### 1.3 데이터 모델

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

## 2. 프로젝트 구조

### 2.1 디렉토리 레이아웃

```
urlshort/
├── cmd/
│   └── server/
│       └── main.go              # 진입점
├── internal/
│   ├── config/
│   │   └── config.go            # 설정
│   ├── domain/
│   │   ├── url.go               # URL 엔티티 및 리포지토리 인터페이스
│   │   └── user.go              # User 엔티티 및 리포지토리 인터페이스
│   ├── handler/
│   │   ├── url_handler.go       # URL HTTP 핸들러
│   │   ├── auth_handler.go      # 인증 HTTP 핸들러
│   │   └── health_handler.go    # 헬스 체크 핸들러
│   ├── middleware/
│   │   ├── auth.go              # JWT 인증
│   │   ├── logging.go           # 요청 로깅
│   │   ├── ratelimit.go         # 요청 제한
│   │   └── recovery.go          # 패닉 복구
│   ├── repository/
│   │   ├── postgres/
│   │   │   ├── url_repo.go      # PostgreSQL URL 리포지토리
│   │   │   └── user_repo.go     # PostgreSQL 사용자 리포지토리
│   │   └── redis/
│   │       └── cache.go         # Redis 캐시
│   ├── service/
│   │   ├── url_service.go       # URL 비즈니스 로직
│   │   └── auth_service.go      # 인증 비즈니스 로직
│   └── server/
│       └── server.go            # HTTP 서버 설정
├── migrations/
│   ├── 001_create_users.sql
│   ├── 002_create_urls.sql
│   └── 003_create_clicks.sql
├── tests/
│   └── integration/
│       └── api_test.go          # 통합 테스트
├── Dockerfile
├── docker-compose.yml
├── go.mod
├── go.sum
└── Makefile
```

### 2.2 진입점

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

## 3. 핵심 도메인

### 3.1 URL 서비스

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

## 4. HTTP 계층

### 4.1 URL 핸들러

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

### 4.2 라우터 설정

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

## 5. 영속성 계층

### 5.1 PostgreSQL 리포지토리

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

### 5.2 데이터베이스 마이그레이션

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

## 6. 테스트

### 6.1 유닛 테스트

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

### 6.2 통합 테스트

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

## 7. 요약

### 핵심 포인트

1. **클린 아키텍처** — 핸들러(HTTP), 서비스(비즈니스 로직), 리포지토리(영속성)를 분리한다.
2. **인터페이스 경계** — 테스트 가능성과 유연성을 위해 소비자 측에서 인터페이스를 정의한다.
3. **미들웨어 체인** — 로깅, 인증, 요청 제한, 복구를 조합 가능한 계층으로 구성한다.
4. **헬스 체크** — 클라우드 배포에 필수이다. 레디니스는 트래픽을 제어하고, 라이브니스는 재시작을 제어한다.
5. **그레이스풀 셧다운** — 트래픽 수신 중단, 요청 처리 완료, 리소스 해제 순으로 진행한다.
6. **포괄적인 테스트** — 모의 객체를 사용한 유닛 테스트, 실제 데이터베이스를 사용한 통합 테스트를 작성한다.
7. **Docker와 CI/CD** — 멀티 스테이지 빌드, 자동화된 테스트, 릴리스 자동화를 구성한다.

### 과정 요약

이 과정은 Go 기초부터 프로덕션 마이크로서비스까지 다루었다:

| 단계 | 레슨 | 핵심 기술 |
|------|-------|-----------|
| 기초 | 01-06 | 타입, 함수, 인터페이스, 에러, 패키지 |
| 동시성 | 07-09 | 고루틴, 채널, 패턴, context |
| 도구 | 10-11 | 테스트, 표준 라이브러리 |
| 웹 | 12-14 | HTTP, REST, 데이터베이스 |
| 고급 | 15-18 | CLI, 제네릭, 리플렉션, 프로파일링 |
| 프로덕션 | 19-22 | Docker, 네트워킹, 클라우드 네이티브, 캡스톤 |

이제 프로덕션에 적합한 Go 애플리케이션을 구축할 수 있는 지식을 갖추었다. 다음 단계는 실습이다 — 프로젝트를 만들고, Go 소스 코드를 읽고, 오픈소스 Go 프로젝트에 기여한다.

---

## 연습 문제

### 연습 1: 마이크로서비스 완성
인증, CRUD, 캐싱, 클릭 분석, 테스트를 포함한 완전한 URL 단축 서비스를 구현한다.

### 연습 2: 기능 추가
다음으로 확장한다: 커스텀 슬러그, QR 코드 생성, 대량 가져오기, 만료 알림, 관리자 대시보드.

### 연습 3: 성능 테스트
`hey` 또는 `wrk`로 부하 테스트를 수행한다. 프로파일링하고 병목 지점을 식별하여 최적화한다. 목표: 초당 10,000건의 리다이렉트.

### 연습 4: 클라우드 배포
클라우드 제공업체에 배포한다: Docker, 관리형 데이터베이스, CDN, TLS, 모니터링, 알림을 포함한다.
