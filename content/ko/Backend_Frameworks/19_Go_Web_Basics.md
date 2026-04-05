# 19. Go 웹 기초

**이전**: [프로젝트: REST API](./18_Project_REST_API.md) | **다음**: [Redis 캐싱 패턴](./20_Redis_Caching_Patterns.md)

**난이도**: ⭐⭐⭐

## 학습 목표

- 백엔드 서비스에서 Go가 강력한 선택인 이유를 이해한다 (동시성, 성능, 단순성)
- Go의 `net/http` 표준 라이브러리를 사용하여 HTTP 서버를 구축한다
- Gin 프레임워크로 라우팅, 미들웨어, 요청 바인딩을 구현한다
- Gin과 Echo 프레임워크를 비교하여 프레임워크 선택에 필요한 정보를 얻는다
- JSON 요청과 응답을 Go 관용적 방식으로 처리한다
- 로깅, 인증, 복구를 위한 미들웨어 패턴을 구현한다
- GORM을 사용하여 SQL 데이터베이스를 통합한다
- 웹 애플리케이션에서 Go 관용적 오류 처리 패턴을 적용한다
- 확립된 Go 프로젝트 구조 관례를 따른다

## 목차

1. [백엔드 개발에 Go를 선택하는 이유](#1-백엔드-개발에-go를-선택하는-이유)
2. [net/http 표준 라이브러리](#2-nethttp-표준-라이브러리)
3. [Gin 프레임워크 기초](#3-gin-프레임워크-기초)
4. [Echo 프레임워크 비교](#4-echo-프레임워크-비교)
5. [요청 처리와 JSON 응답](#5-요청-처리와-json-응답)
6. [미들웨어 패턴](#6-미들웨어-패턴)
7. [GORM을 이용한 데이터베이스 통합](#7-gorm을-이용한-데이터베이스-통합)
8. [오류 처리 패턴](#8-오류-처리-패턴)
9. [프로젝트 구조 관례](#9-프로젝트-구조-관례)
10. [Go 웹 애플리케이션 테스트](#10-go-웹-애플리케이션-테스트)
11. [연습 문제](#11-연습-문제)

---

## 1. 백엔드 개발에 Go를 선택하는 이유

Go(Golang)는 백엔드 서비스 구축을 위한 가장 인기 있는 언어 중 하나가 되었다. Google, Uber, Dropbox, Twitch와 같은 기업들이 고성능 동시성 시스템에 Go를 활용하고 있다.

### 주요 장점

**동시성 모델(Concurrency model)**: Go의 고루틴(goroutine)과 채널(channel)은 동시성 프로그래밍을 접근하기 쉽게 만든다. 고루틴은 대략 2 KB의 스택 공간을 사용하며(OS 스레드당 약 1 MB에 비해), 수백만 개의 동시 작업이 가능하다.

```
OS Thread Model:              Goroutine Model:
┌────────────────┐           ┌────────────────┐
│  OS Thread 1   │           │  OS Thread 1   │
│  (1-8 MB stack)│           │  (manages many) │
│  ┌──────────┐  │           │  ┌──┐┌──┐┌──┐  │
│  │ 1 handler│  │           │  │g1││g2││g3│  │
│  └──────────┘  │           │  └──┘└──┘└──┘  │
├────────────────┤           │  ┌──┐┌──┐┌──┐  │
│  OS Thread 2   │           │  │g4││g5││g6│  │
│  ┌──────────┐  │           │  └──┘└──┘└──┘  │
│  │ 1 handler│  │           └────────────────┘
│  └──────────┘  │           Each goroutine: ~2KB
└────────────────┘           Thousands on one thread
```

**성능(Performance)**: Go는 가상 머신 오버헤드 없이 네이티브 머신 코드로 컴파일된다. 일반적인 Go 웹 서비스는 동등한 Python/Node.js 서비스 대비 10-50배의 처리량을 보여준다.

**단순성(Simplicity)**: Go는 약 50개의 키워드로 구성된 작은 언어 명세를 가진다. 표준 라이브러리가 광범위하며, 생태계는 "마법"보다 명시적인 코드를 선호한다.

**빠른 컴파일**: 대규모 Go 프로젝트도 수 초 내에 컴파일되어 빠른 개발 주기를 가능하게 한다.

**단일 바이너리 배포**: Go는 런타임 의존성이 없는 정적 링크된 바이너리를 생성한다. 단일 파일을 복사하는 것만으로 배포가 완료된다.

**내장 도구**: `go fmt`, `go vet`, `go test`, `go doc`이 표준 도구 체인에 포함되어 있다.

### Go가 탁월한 경우

| 사용 사례 | Go가 적합한 이유 |
|---|---|
| 마이크로서비스(Microservices) | 작은 바이너리, 빠른 시작, 낮은 메모리 |
| API 게이트웨이(API gateways) | 높은 동시성, 낮은 지연 시간 |
| CLI 도구(CLI tools) | 단일 바이너리, 크로스 컴파일 |
| 데이터 파이프라인(Data pipelines) | 병렬 처리를 위한 고루틴 |
| DevOps 도구(DevOps tooling) | Docker, Kubernetes, Terraform이 Go로 작성됨 |

### 대안을 고려해야 할 경우

- 빠른 프로토타이핑(prototyping) — Python/Node.js가 더 빠르게 반복 개발 가능
- ORM 중심의 CRUD 앱 — Django/Rails가 더 많은 기능을 기본 제공
- 데이터 과학과 ML — Python 생태계가 더 강력

---

## 2. net/http 표준 라이브러리

Go의 `net/http` 패키지는 프로덕션 수준의 품질을 가진다. 많은 팀이 프레임워크 없이 이 패키지만으로 개발한다.

### 기본 HTTP 서버

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"
)

func main() {
    mux := http.NewServeMux()

    mux.HandleFunc("GET /", handleHome)
    mux.HandleFunc("GET /health", handleHealth)
    mux.HandleFunc("GET /api/users/{id}", handleGetUser)
    mux.HandleFunc("POST /api/users", handleCreateUser)

    server := &http.Server{
        Addr:         ":8080",
        Handler:      mux,
        ReadTimeout:  10 * time.Second,
        WriteTimeout: 10 * time.Second,
        IdleTimeout:  120 * time.Second,
    }

    log.Printf("Server starting on %s", server.Addr)
    if err := server.ListenAndServe(); err != nil {
        log.Fatalf("Server failed: %v", err)
    }
}

func handleHome(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintln(w, "Welcome to the Go API")
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{
        "status": "healthy",
        "time":   time.Now().UTC().Format(time.RFC3339),
    })
}
```

### 경로 매개변수 (Go 1.22+)

Go 1.22에서 `http.ServeMux`에 패턴 매칭이 도입되었다:

```go
func handleGetUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id")
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{
        "id":   id,
        "name": "User " + id,
    })
}
```

### 요청 본문 읽기

```go
type CreateUserRequest struct {
    Name  string `json:"name"`
    Email string `json:"email"`
}

func handleCreateUser(w http.ResponseWriter, r *http.Request) {
    var req CreateUserRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, `{"error": "invalid JSON"}`, http.StatusBadRequest)
        return
    }
    defer r.Body.Close()

    if req.Name == "" || req.Email == "" {
        http.Error(w, `{"error": "name and email required"}`, http.StatusBadRequest)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(map[string]interface{}{
        "id":    42,
        "name":  req.Name,
        "email": req.Email,
    })
}
```

### net/http를 이용한 커스텀 미들웨어

```go
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        log.Printf("Started %s %s", r.Method, r.URL.Path)

        next.ServeHTTP(w, r)

        log.Printf("Completed %s %s in %v", r.Method, r.URL.Path, time.Since(start))
    })
}

func recoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("Panic recovered: %v", err)
                http.Error(w, `{"error": "internal server error"}`, http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// 미들웨어 체이닝
func main() {
    mux := http.NewServeMux()
    // ... 핸들러 등록 ...

    handler := loggingMiddleware(recoveryMiddleware(mux))

    http.ListenAndServe(":8080", handler)
}
```

---

## 3. Gin 프레임워크 기초

[Gin](https://github.com/gin-gonic/gin)은 Go에서 가장 인기 있는 웹 프레임워크로, 빠른 속도와 간결한 API로 유명하다.

### 설치와 Hello World

```bash
go mod init myapp
go get -u github.com/gin-gonic/gin
```

```go
package main

import (
    "net/http"
    "github.com/gin-gonic/gin"
)

func main() {
    r := gin.Default() // Logger와 Recovery 미들웨어 포함

    r.GET("/", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "message": "Hello, World!",
        })
    })

    r.Run(":8080")
}
```

### 라우팅(Routing)

```go
func setupRoutes(r *gin.Engine) {
    // 기본 라우트
    r.GET("/ping", handlePing)
    r.POST("/users", createUser)
    r.GET("/users/:id", getUser)
    r.PUT("/users/:id", updateUser)
    r.DELETE("/users/:id", deleteUser)

    // 라우트 그룹
    api := r.Group("/api/v1")
    {
        api.GET("/products", listProducts)
        api.POST("/products", createProduct)

        // 미들웨어를 포함한 중첩 그룹
        admin := api.Group("/admin")
        admin.Use(authMiddleware())
        {
            admin.GET("/stats", getStats)
            admin.DELETE("/users/:id", deleteUser)
        }
    }

    // 쿼리 매개변수: /search?q=go&page=1
    r.GET("/search", func(c *gin.Context) {
        query := c.DefaultQuery("q", "")
        page := c.DefaultQuery("page", "1")
        c.JSON(http.StatusOK, gin.H{"query": query, "page": page})
    })
}
```

### 요청 바인딩과 검증(Validation)

Gin은 구조체 태그(struct tag)를 사용하여 자동 요청 바인딩과 검증을 수행한다:

```go
type CreateProductRequest struct {
    Name     string  `json:"name" binding:"required,min=1,max=200"`
    Price    float64 `json:"price" binding:"required,gt=0"`
    Category string  `json:"category" binding:"required,oneof=electronics books clothing"`
    SKU      string  `json:"sku" binding:"required,alphanum,len=8"`
}

func createProduct(c *gin.Context) {
    var req CreateProductRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }

    // req가 검증되고 값이 채워진 상태
    c.JSON(http.StatusCreated, gin.H{
        "id":       1,
        "name":     req.Name,
        "price":    req.Price,
        "category": req.Category,
    })
}

// 경로 매개변수와 쿼리 매개변수 바인딩
type GetProductsQuery struct {
    Page     int    `form:"page" binding:"min=1"`
    PageSize int    `form:"page_size" binding:"min=1,max=100"`
    Sort     string `form:"sort" binding:"oneof=name price created_at"`
}

type URIParams struct {
    ID uint `uri:"id" binding:"required,min=1"`
}

func getUser(c *gin.Context) {
    var uri URIParams
    if err := c.ShouldBindUri(&uri); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "invalid user ID"})
        return
    }
    // uri.ID 사용 ...
}
```

### Gin 컨텍스트(Context)

`gin.Context`는 모든 핸들러의 핵심으로, 요청 데이터, 응답 메서드, 흐름 제어를 담당한다:

```go
func exampleHandler(c *gin.Context) {
    // 요청 데이터
    id := c.Param("id")                          // 경로 매개변수
    name := c.Query("name")                       // 쿼리 매개변수
    token := c.GetHeader("Authorization")          // 요청 헤더

    // 하위 미들웨어/핸들러를 위한 값 설정
    c.Set("user_id", 42)
    userID, exists := c.Get("user_id")

    // 응답 메서드
    c.JSON(http.StatusOK, gin.H{"id": id})         // JSON 응답
    c.String(http.StatusOK, "Hello %s", name)       // 일반 텍스트
    c.Data(http.StatusOK, "text/csv", csvBytes)     // 원시 바이트
    c.File("./path/to/file.pdf")                    // 파일 제공

    // 요청 체인 중단
    c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "access denied"})
}
```

---

## 4. Echo 프레임워크 비교

[Echo](https://echo.labstack.com/)는 또 다른 인기 있는 Go 웹 프레임워크이다. Gin과의 비교를 살펴보자.

### Echo Hello World

```go
package main

import (
    "net/http"
    "github.com/labstack/echo/v4"
    "github.com/labstack/echo/v4/middleware"
)

func main() {
    e := echo.New()
    e.Use(middleware.Logger())
    e.Use(middleware.Recover())

    e.GET("/", func(c echo.Context) error {
        return c.JSON(http.StatusOK, map[string]string{
            "message": "Hello, World!",
        })
    })

    e.Logger.Fatal(e.Start(":8080"))
}
```

### 기능 비교

| 기능 | Gin | Echo |
|---|---|---|
| 핸들러 시그니처 | `func(c *gin.Context)` | `func(c echo.Context) error` |
| 오류 처리 | 수동 (반환값 없음) | `error` 반환 (중앙 집중식) |
| 바인딩 | `c.ShouldBindJSON(&s)` | `c.Bind(&s)` |
| 경로 매개변수 | `c.Param("id")` | `c.Param("id")` |
| 미들웨어 | `r.Use(fn)` | `e.Use(fn)` |
| 라우트 그룹 | `r.Group("/api")` | `e.Group("/api")` |
| 성능 | ~65,000 req/s | ~63,000 req/s |
| GitHub 스타 | ~80k | ~30k |

### Echo 오류 처리 (핵심 차별점)

Echo의 핸들러 시그니처는 `error`를 반환하여 중앙 집중식 오류 처리가 가능하다:

```go
// Echo 커스텀 오류 핸들러
func customErrorHandler(err error, c echo.Context) {
    code := http.StatusInternalServerError
    message := "Internal Server Error"

    if he, ok := err.(*echo.HTTPError); ok {
        code = he.Code
        message = he.Message.(string)
    }

    c.JSON(code, map[string]string{"error": message})
}

func main() {
    e := echo.New()
    e.HTTPErrorHandler = customErrorHandler

    e.GET("/users/:id", func(c echo.Context) error {
        id := c.Param("id")
        user, err := findUser(id)
        if err != nil {
            return echo.NewHTTPError(http.StatusNotFound, "user not found")
        }
        return c.JSON(http.StatusOK, user)
    })
}
```

### 어떤 것을 선택할까?

- **Gin**: 더 큰 커뮤니티, 더 많은 서드파티 미들웨어, 잘 확립된 생태계
- **Echo**: 더 깔끔한 오류 처리, 약간 더 나은 문서화, 내장 HTTPS 지원
- 둘 다 우수하며, 팀의 선호도에 따라 선택하면 된다

---

## 5. 요청 처리와 JSON 응답

### 일관된 응답 엔벨로프(Envelope)

API 전반에 걸쳐 표준 응답 형식을 정의한다:

```go
// response.go
package response

import (
    "net/http"
    "github.com/gin-gonic/gin"
)

type APIResponse struct {
    Success bool        `json:"success"`
    Data    interface{} `json:"data,omitempty"`
    Error   *APIError   `json:"error,omitempty"`
    Meta    *Meta       `json:"meta,omitempty"`
}

type APIError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
}

type Meta struct {
    Page       int `json:"page"`
    PageSize   int `json:"page_size"`
    TotalCount int `json:"total_count"`
    TotalPages int `json:"total_pages"`
}

func Success(c *gin.Context, status int, data interface{}) {
    c.JSON(status, APIResponse{Success: true, Data: data})
}

func SuccessWithMeta(c *gin.Context, data interface{}, meta Meta) {
    c.JSON(http.StatusOK, APIResponse{
        Success: true,
        Data:    data,
        Meta:    &meta,
    })
}

func Error(c *gin.Context, status int, code, message string) {
    c.JSON(status, APIResponse{
        Success: false,
        Error:   &APIError{Code: code, Message: message},
    })
}
```

### 다양한 콘텐츠 유형 처리

```go
func handleUpload(c *gin.Context) {
    // 멀티파트 폼
    file, err := c.FormFile("avatar")
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "no file uploaded"})
        return
    }

    dst := fmt.Sprintf("./uploads/%s", file.Filename)
    if err := c.SaveUploadedFile(file, dst); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to save file"})
        return
    }

    c.JSON(http.StatusOK, gin.H{
        "filename": file.Filename,
        "size":     file.Size,
    })
}

func handleFormData(c *gin.Context) {
    // URL 인코딩된 폼 데이터
    name := c.PostForm("name")
    email := c.PostForm("email")
    c.JSON(http.StatusOK, gin.H{"name": name, "email": email})
}
```

### 스트리밍 응답

```go
func handleStream(c *gin.Context) {
    c.Stream(func(w io.Writer) bool {
        for i := 0; i < 10; i++ {
            c.SSEvent("message", gin.H{"count": i})
            time.Sleep(1 * time.Second)
        }
        return false
    })
}
```

---

## 6. 미들웨어 패턴

### 인증 미들웨어(Authentication Middleware)

```go
func authMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        token := c.GetHeader("Authorization")
        if token == "" {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "error": "authorization header required",
            })
            return
        }

        // "Bearer " 접두사 제거
        if len(token) > 7 && token[:7] == "Bearer " {
            token = token[7:]
        }

        claims, err := validateJWT(token)
        if err != nil {
            c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
                "error": "invalid or expired token",
            })
            return
        }

        // 하위 핸들러를 위해 사용자 정보 저장
        c.Set("user_id", claims.UserID)
        c.Set("user_role", claims.Role)
        c.Next()
    }
}

// 역할 기반 인가(Role-based authorization)
func requireRole(roles ...string) gin.HandlerFunc {
    return func(c *gin.Context) {
        userRole, exists := c.Get("user_role")
        if !exists {
            c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "no role found"})
            return
        }

        for _, role := range roles {
            if userRole == role {
                c.Next()
                return
            }
        }

        c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "insufficient permissions"})
    }
}
```

### 속도 제한 미들웨어(Rate Limiting Middleware)

```go
import "golang.org/x/time/rate"

func rateLimitMiddleware(rps float64, burst int) gin.HandlerFunc {
    limiter := rate.NewLimiter(rate.Limit(rps), burst)
    return func(c *gin.Context) {
        if !limiter.Allow() {
            c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
                "error": "rate limit exceeded",
            })
            return
        }
        c.Next()
    }
}

// 클라이언트별 속도 제한 (맵 사용)
func perClientRateLimiter(rps float64, burst int) gin.HandlerFunc {
    clients := make(map[string]*rate.Limiter)
    var mu sync.Mutex

    return func(c *gin.Context) {
        ip := c.ClientIP()

        mu.Lock()
        limiter, exists := clients[ip]
        if !exists {
            limiter = rate.NewLimiter(rate.Limit(rps), burst)
            clients[ip] = limiter
        }
        mu.Unlock()

        if !limiter.Allow() {
            c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
                "error": "rate limit exceeded",
            })
            return
        }
        c.Next()
    }
}
```

### CORS 미들웨어

```go
func corsMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Header("Access-Control-Allow-Origin", "*")
        c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        c.Header("Access-Control-Max-Age", "86400")

        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(http.StatusNoContent)
            return
        }

        c.Next()
    }
}
```

### 요청 ID 미들웨어(Request ID Middleware)

```go
import "github.com/google/uuid"

func requestIDMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        requestID := c.GetHeader("X-Request-ID")
        if requestID == "" {
            requestID = uuid.New().String()
        }
        c.Set("request_id", requestID)
        c.Header("X-Request-ID", requestID)
        c.Next()
    }
}
```

---

## 7. GORM을 이용한 데이터베이스 통합

[GORM](https://gorm.io/)은 Go에서 가장 인기 있는 ORM이다. PostgreSQL, MySQL, SQLite, SQL Server를 지원한다.

### 설정

```bash
go get -u gorm.io/gorm
go get -u gorm.io/driver/postgres
```

### 모델 정의

```go
package models

import (
    "time"
    "gorm.io/gorm"
)

type User struct {
    ID        uint           `gorm:"primaryKey" json:"id"`
    CreatedAt time.Time      `json:"created_at"`
    UpdatedAt time.Time      `json:"updated_at"`
    DeletedAt gorm.DeletedAt `gorm:"index" json:"-"`
    Name      string         `gorm:"size:100;not null" json:"name"`
    Email     string         `gorm:"uniqueIndex;size:255;not null" json:"email"`
    Password  string         `gorm:"size:255;not null" json:"-"`
    Role      string         `gorm:"size:20;default:user" json:"role"`
    Posts     []Post         `gorm:"foreignKey:AuthorID" json:"posts,omitempty"`
}

type Post struct {
    ID        uint           `gorm:"primaryKey" json:"id"`
    CreatedAt time.Time      `json:"created_at"`
    UpdatedAt time.Time      `json:"updated_at"`
    DeletedAt gorm.DeletedAt `gorm:"index" json:"-"`
    Title     string         `gorm:"size:200;not null" json:"title"`
    Content   string         `gorm:"type:text;not null" json:"content"`
    Published bool           `gorm:"default:false" json:"published"`
    AuthorID  uint           `gorm:"not null" json:"author_id"`
    Author    User           `gorm:"foreignKey:AuthorID" json:"author,omitempty"`
    Tags      []Tag          `gorm:"many2many:post_tags;" json:"tags,omitempty"`
}

type Tag struct {
    ID   uint   `gorm:"primaryKey" json:"id"`
    Name string `gorm:"uniqueIndex;size:50" json:"name"`
}
```

### 데이터베이스 연결

```go
package database

import (
    "fmt"
    "log"
    "os"
    "time"

    "gorm.io/driver/postgres"
    "gorm.io/gorm"
    "gorm.io/gorm/logger"
)

var DB *gorm.DB

func Connect() {
    dsn := fmt.Sprintf(
        "host=%s user=%s password=%s dbname=%s port=%s sslmode=disable",
        os.Getenv("DB_HOST"),
        os.Getenv("DB_USER"),
        os.Getenv("DB_PASSWORD"),
        os.Getenv("DB_NAME"),
        os.Getenv("DB_PORT"),
    )

    var err error
    DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{
        Logger: logger.Default.LogMode(logger.Info),
    })
    if err != nil {
        log.Fatalf("Failed to connect to database: %v", err)
    }

    // 커넥션 풀 설정
    sqlDB, _ := DB.DB()
    sqlDB.SetMaxIdleConns(10)
    sqlDB.SetMaxOpenConns(100)
    sqlDB.SetConnMaxLifetime(time.Hour)

    log.Println("Database connected successfully")
}

func Migrate() {
    DB.AutoMigrate(&User{}, &Post{}, &Tag{})
    log.Println("Database migration completed")
}
```

### CRUD 작업

```go
package repository

import "gorm.io/gorm"

type PostRepository struct {
    db *gorm.DB
}

func NewPostRepository(db *gorm.DB) *PostRepository {
    return &PostRepository{db: db}
}

func (r *PostRepository) Create(post *Post) error {
    return r.db.Create(post).Error
}

func (r *PostRepository) FindByID(id uint) (*Post, error) {
    var post Post
    err := r.db.Preload("Author").Preload("Tags").First(&post, id).Error
    if err != nil {
        return nil, err
    }
    return &post, nil
}

func (r *PostRepository) List(page, pageSize int, published *bool) ([]Post, int64, error) {
    var posts []Post
    var total int64

    query := r.db.Model(&Post{})
    if published != nil {
        query = query.Where("published = ?", *published)
    }

    query.Count(&total)

    err := query.
        Preload("Author").
        Preload("Tags").
        Offset((page - 1) * pageSize).
        Limit(pageSize).
        Order("created_at DESC").
        Find(&posts).Error

    return posts, total, err
}

func (r *PostRepository) Update(post *Post) error {
    return r.db.Save(post).Error
}

func (r *PostRepository) Delete(id uint) error {
    return r.db.Delete(&Post{}, id).Error // 소프트 삭제
}

// 트랜잭션 예제
func (r *PostRepository) CreateWithTags(post *Post, tagNames []string) error {
    return r.db.Transaction(func(tx *gorm.DB) error {
        if err := tx.Create(post).Error; err != nil {
            return err
        }

        for _, name := range tagNames {
            var tag Tag
            tx.FirstOrCreate(&tag, Tag{Name: name})
            if err := tx.Model(post).Association("Tags").Append(&tag); err != nil {
                return err
            }
        }

        return nil
    })
}
```

---

## 8. 오류 처리 패턴

### 커스텀 오류 타입

```go
package apperror

import "fmt"

type AppError struct {
    Code    int    `json:"-"`
    Type    string `json:"type"`
    Message string `json:"message"`
    Detail  string `json:"detail,omitempty"`
}

func (e *AppError) Error() string {
    return fmt.Sprintf("%s: %s", e.Type, e.Message)
}

func NotFound(resource string, id interface{}) *AppError {
    return &AppError{
        Code:    404,
        Type:    "NOT_FOUND",
        Message: fmt.Sprintf("%s not found", resource),
        Detail:  fmt.Sprintf("No %s with ID %v", resource, id),
    }
}

func BadRequest(message string) *AppError {
    return &AppError{Code: 400, Type: "BAD_REQUEST", Message: message}
}

func Unauthorized(message string) *AppError {
    return &AppError{Code: 401, Type: "UNAUTHORIZED", Message: message}
}

func Forbidden(message string) *AppError {
    return &AppError{Code: 403, Type: "FORBIDDEN", Message: message}
}

func Internal(err error) *AppError {
    return &AppError{
        Code:    500,
        Type:    "INTERNAL_ERROR",
        Message: "An internal error occurred",
        Detail:  err.Error(),
    }
}
```

### 오류 처리 미들웨어

```go
func errorHandler() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Next()

        // 요청 처리 중 설정된 오류 확인
        if len(c.Errors) > 0 {
            err := c.Errors.Last().Err
            switch e := err.(type) {
            case *apperror.AppError:
                c.JSON(e.Code, gin.H{
                    "success": false,
                    "error":   e,
                })
            default:
                c.JSON(http.StatusInternalServerError, gin.H{
                    "success": false,
                    "error": gin.H{
                        "type":    "INTERNAL_ERROR",
                        "message": "An unexpected error occurred",
                    },
                })
            }
        }
    }
}

// 핸들러에서의 사용
func getPost(c *gin.Context) {
    id, err := strconv.ParseUint(c.Param("id"), 10, 32)
    if err != nil {
        c.Error(apperror.BadRequest("invalid post ID"))
        return
    }

    post, err := postRepo.FindByID(uint(id))
    if err != nil {
        if errors.Is(err, gorm.ErrRecordNotFound) {
            c.Error(apperror.NotFound("post", id))
        } else {
            c.Error(apperror.Internal(err))
        }
        return
    }

    c.JSON(http.StatusOK, gin.H{"success": true, "data": post})
}
```

### 컨텍스트를 포함한 오류 래핑(Wrapping)

```go
import "fmt"

func (s *PostService) Publish(postID, userID uint) error {
    post, err := s.repo.FindByID(postID)
    if err != nil {
        return fmt.Errorf("publish: finding post %d: %w", postID, err)
    }

    if post.AuthorID != userID {
        return fmt.Errorf("publish: user %d is not author of post %d: %w",
            userID, postID, apperror.Forbidden("not the author"))
    }

    post.Published = true
    if err := s.repo.Update(post); err != nil {
        return fmt.Errorf("publish: updating post %d: %w", postID, err)
    }

    return nil
}
```

---

## 9. 프로젝트 구조 관례

### 표준 Go 프로젝트 레이아웃

```
myapp/
├── cmd/
│   └── server/
│       └── main.go              # 애플리케이션 진입점
├── internal/                    # 비공개 애플리케이션 코드
│   ├── config/
│   │   └── config.go            # 설정 로딩
│   ├── database/
│   │   └── database.go          # 데이터베이스 연결
│   ├── handler/
│   │   ├── user_handler.go      # HTTP 핸들러 (컨트롤러)
│   │   └── post_handler.go
│   ├── middleware/
│   │   ├── auth.go
│   │   ├── cors.go
│   │   └── logging.go
│   ├── model/
│   │   ├── user.go              # 데이터베이스 모델
│   │   └── post.go
│   ├── repository/
│   │   ├── user_repo.go         # 데이터 접근 계층
│   │   └── post_repo.go
│   ├── service/
│   │   ├── user_service.go      # 비즈니스 로직
│   │   └── post_service.go
│   └── router/
│       └── router.go            # 라우트 정의
├── pkg/                         # 공개 재사용 가능 패키지
│   ├── apperror/
│   │   └── errors.go
│   └── response/
│       └── response.go
├── migrations/                  # SQL 마이그레이션 파일
├── docs/                        # API 문서
├── go.mod
├── go.sum
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

### 설정 관리(Configuration Management)

```go
// internal/config/config.go
package config

import (
    "log"
    "os"
    "strconv"
)

type Config struct {
    Port        string
    DatabaseURL string
    JWTSecret   string
    LogLevel    string
    Environment string
}

func Load() *Config {
    return &Config{
        Port:        getEnv("PORT", "8080"),
        DatabaseURL: getEnv("DATABASE_URL", "postgres://localhost:5432/myapp?sslmode=disable"),
        JWTSecret:   getEnv("JWT_SECRET", "change-me-in-production"),
        LogLevel:    getEnv("LOG_LEVEL", "info"),
        Environment: getEnv("ENVIRONMENT", "development"),
    }
}

func getEnv(key, fallback string) string {
    if value, ok := os.LookupEnv(key); ok {
        return value
    }
    return fallback
}

func getEnvInt(key string, fallback int) int {
    if value, ok := os.LookupEnv(key); ok {
        if i, err := strconv.Atoi(value); err == nil {
            return i
        }
    }
    return fallback
}
```

### 공통 작업을 위한 Makefile

```makefile
.PHONY: run build test lint migrate

run:
	go run cmd/server/main.go

build:
	CGO_ENABLED=0 go build -o bin/server cmd/server/main.go

test:
	go test ./... -v -cover

lint:
	golangci-lint run

migrate:
	go run cmd/migrate/main.go

docker-build:
	docker build -t myapp .

docker-run:
	docker-compose up -d
```

### Dockerfile

```dockerfile
# 빌드 단계
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /server cmd/server/main.go

# 실행 단계
FROM alpine:3.19
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /server .
EXPOSE 8080
CMD ["./server"]
```

---

## 10. Go 웹 애플리케이션 테스트

Go 표준 라이브러리에는 외부 프레임워크 없이 HTTP 핸들러를 테스트하는 데 필요한 모든 것이 포함되어 있다.

### 테이블 기반 테스트(Table-Driven Tests)

관용적인 Go 테스트 패턴은 관련 케이스를 구조체 슬라이스(slice of structs)로 그룹화한다:

```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name   string
        a, b   int
        want   int
    }{
        {"positive", 1, 2, 3},
        {"negative", -1, -2, -3},
        {"zero", 0, 5, 5},
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            if got := Add(tt.a, tt.b); got != tt.want {
                t.Errorf("Add(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.want)
            }
        })
    }
}
```

### httptest 패키지

`net/http/httptest`를 사용하면 실제 서버를 시작하지 않고 핸들러를 직접 호출할 수 있다:

```go
import (
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"
)

func TestHandleHealth(t *testing.T) {
    req := httptest.NewRequest(http.MethodGet, "/health", nil)
    rec := httptest.NewRecorder()

    handleHealth(rec, req)

    if rec.Code != http.StatusOK {
        t.Fatalf("expected 200, got %d", rec.Code)
    }
    var body map[string]string
    json.NewDecoder(rec.Body).Decode(&body)
    if body["status"] != "healthy" {
        t.Errorf("unexpected status: %s", body["status"])
    }
}
```

Gin 핸들러의 경우 엔진을 `httptest.NewRecorder()`로 래핑한다:

```go
func TestCreateProduct(t *testing.T) {
    gin.SetMode(gin.TestMode)
    r := gin.New()
    r.POST("/products", createProduct)

    body := `{"name":"Widget","price":9.99,"category":"electronics","sku":"ABCD1234"}`
    req := httptest.NewRequest(http.MethodPost, "/products", strings.NewReader(body))
    req.Header.Set("Content-Type", "application/json")
    rec := httptest.NewRecorder()

    r.ServeHTTP(rec, req)

    if rec.Code != http.StatusCreated {
        t.Fatalf("expected 201, got %d: %s", rec.Code, rec.Body.String())
    }
}
```

### 어서션을 위한 testify

[testify](https://github.com/stretchr/testify)는 `assert`와 `require`로 보일러플레이트(boilerplate)를 줄인다:

```go
import "github.com/stretchr/testify/assert"

func TestUserService(t *testing.T) {
    svc := NewUserService(setupTestDB(t))
    user, err := svc.Create("alice@example.com", "Alice")

    assert.NoError(t, err)
    assert.Equal(t, "Alice", user.Name)
    assert.NotZero(t, user.ID)
}
```

필드에 접근하기 전 nil 포인터 확인처럼 실패 시 테스트를 즉시 중단해야 하는 경우 `assert` 대신 `require`를 사용한다.

---

## 11. 연습 문제

### 연습 1: net/http를 이용한 기본 API

Go 표준 라이브러리만 사용하여 북마크 관리 API를 구축하라:
- `GET /bookmarks` — 모든 북마크 목록 조회 (인메모리 슬라이스)
- `POST /bookmarks` — 북마크 추가 (제목, URL, 태그)
- `GET /bookmarks/{id}` — 단일 북마크 조회
- `DELETE /bookmarks/{id}` — 북마크 삭제
- 메서드, 경로, 소요 시간을 출력하는 로깅 미들웨어를 추가하라

```go
// 시작 코드
package main

import (
    "encoding/json"
    "log"
    "net/http"
    "sync"
    "time"
)

type Bookmark struct {
    ID        int       `json:"id"`
    Title     string    `json:"title"`
    URL       string    `json:"url"`
    Tags      []string  `json:"tags"`
    CreatedAt time.Time `json:"created_at"`
}

var (
    bookmarks = make([]Bookmark, 0)
    nextID    = 1
    mu        sync.Mutex
)

func main() {
    mux := http.NewServeMux()

    // TODO: 라우트 등록
    // TODO: 로깅 미들웨어 추가
    // TODO: :8080에서 서버 시작

    log.Fatal(http.ListenAndServe(":8080", mux))
}
```

### 연습 2: Gin을 이용한 CRUD API와 검증

Gin으로 작업 관리 API를 구축하라:
- 작업에 대한 전체 CRUD 구현 (제목, 설명, 상태, 우선순위, 마감일)
- 검증 태그를 포함한 구조체 바인딩 사용
- 라우트 그룹 추가: 공개 라우트와 인증된 라우트
- 간단한 토큰 기반 인증 미들웨어 구현
- 응답 헬퍼를 사용하여 일관된 JSON 응답 반환

### 연습 3: GORM 리포지토리 패턴

도서관 시스템을 위한 리포지토리를 생성하라:
- 모델: `Book`, `Author`, `Genre` (다대다 관계 포함)
- 리포지토리 메서드: `Create`, `FindByID`, `Search` (제목/저자별), `ListByGenre`, `Delete`
- 새 저자와 함께 책을 생성하기 위한 트랜잭션 사용
- 목록 작업에 페이지네이션 추가
- 인메모리 SQLite 데이터베이스를 사용한 테이블 기반 테스트 작성

```go
// 테스트 시작 코드
func TestPostRepository_Create(t *testing.T) {
    db := setupTestDB(t)
    repo := NewPostRepository(db)

    tests := []struct {
        name    string
        post    Post
        wantErr bool
    }{
        {
            name:    "valid post",
            post:    Post{Title: "Test", Content: "Body", AuthorID: 1},
            wantErr: false,
        },
        {
            name:    "missing title",
            post:    Post{Content: "Body", AuthorID: 1},
            wantErr: true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := repo.Create(&tt.post)
            if (err != nil) != tt.wantErr {
                t.Errorf("Create() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### 연습 4: 미들웨어 체인

다음을 포함하는 Gin 애플리케이션용 미들웨어 체인을 구축하라:
1. 요청 ID 생성 (UUID)
2. 구조화된 JSON 로깅 (메서드, 경로, 상태, 소요 시간, 요청 ID)
3. 오류 로깅을 포함한 패닉 복구
4. 설정 가능한 출처를 가진 CORS
5. 속도 제한 (IP당 10 req/s)

`curl` 또는 Go 테스트 클라이언트를 사용하여 동시 요청을 보내 체인을 테스트하라.

---

## 참고 자료

- [Go Documentation](https://go.dev/doc/)
- [Effective Go](https://go.dev/doc/effective_go)
- [Gin Documentation](https://gin-gonic.com/docs/)
- [Echo Documentation](https://echo.labstack.com/docs)
- [GORM Documentation](https://gorm.io/docs/)
- [Standard Go Project Layout](https://github.com/golang-standards/project-layout)

---

**이전**: [프로젝트: REST API](./18_Project_REST_API.md) | **다음**: [Redis 캐싱 패턴](./20_Redis_Caching_Patterns.md)
