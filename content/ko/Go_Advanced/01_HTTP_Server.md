# 12. HTTP 서버

**이전**: [표준 라이브러리](../Go_Basics/11_Standard_Library.md) | **다음**: [REST API](./02_REST_API.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. `net/http`를 사용하여 HTTP 서버를 구축한다
2. `http.ServeMux` (Go 1.22+)를 사용하여 요청 라우팅을 구현한다
3. 로깅, 복구, 인증을 위한 미들웨어를 작성한다
4. 정적 파일과 템플릿을 처리한다
5. 우아한 종료를 구현한다

---

Go의 `net/http` 패키지는 프로덕션에서 사용할 수 있는 HTTP 서버를 기본 제공한다 — 프레임워크가 필요 없다. Go 1.22의 향상된 라우팅으로, 표준 라이브러리가 이전에 서드파티 라우터가 필요했던 경로 매개변수와 메서드 기반 라우팅을 처리한다.

## 목차
1. [기본 서버](#1-기본-서버)
2. [향상된 라우팅 (Go 1.22+)](#2-향상된-라우팅-go-122)
3. [Handler와 HandlerFunc](#3-handler와-handlerfunc)
4. [미들웨어](#4-미들웨어)
5. [템플릿과 정적 파일](#5-템플릿과-정적-파일)
6. [우아한 종료](#6-우아한-종료)
7. [요약](#7-요약)

---

## 1. 기본 서버

### 1.1 Hello World 서버

```go
package main

import (
    "fmt"
    "log"
    "net/http"
)

func main() {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello, %s!", r.URL.Path[1:])
    })

    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.WriteHeader(http.StatusOK)
        fmt.Fprint(w, `{"status": "ok"}`)
    })

    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### 1.2 http.Server 구성

```go
func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("/", homeHandler)

    server := &http.Server{
        Addr:         ":8080",
        Handler:      mux,
        ReadTimeout:  5 * time.Second,
        WriteTimeout: 10 * time.Second,
        IdleTimeout:  120 * time.Second,
        MaxHeaderBytes: 1 << 20, // 1 MB
    }

    log.Println("Server starting on :8080")
    log.Fatal(server.ListenAndServe())
}
```

---

## 2. 향상된 라우팅 (Go 1.22+)

### 2.1 메서드 기반 및 경로 매개변수

```go
func main() {
    mux := http.NewServeMux()

    // 메서드 기반 라우팅
    mux.HandleFunc("GET /users", listUsers)
    mux.HandleFunc("POST /users", createUser)

    // {name}을 사용한 경로 매개변수
    mux.HandleFunc("GET /users/{id}", getUser)
    mux.HandleFunc("PUT /users/{id}", updateUser)
    mux.HandleFunc("DELETE /users/{id}", deleteUser)

    // 와일드카드 — 나머지 경로를 매칭
    mux.HandleFunc("GET /files/{path...}", serveFile)

    // 후행 슬래시와 정확한 매칭
    mux.HandleFunc("GET /api/", apiIndex)      // /api/만 매칭
    mux.HandleFunc("GET /api/{rest...}", apiCatchAll)

    log.Fatal(http.ListenAndServe(":8080", mux))
}

func getUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id") // 경로 매개변수 추출
    fmt.Fprintf(w, "User ID: %s", id)
}

func serveFile(w http.ResponseWriter, r *http.Request) {
    path := r.PathValue("path") // 와일드카드 값
    fmt.Fprintf(w, "File path: %s", path)
}
```

### 2.2 우선순위 규칙

```go
mux := http.NewServeMux()

// 더 구체적인 패턴이 우선
mux.HandleFunc("GET /users/me", getCurrentUser)    // /users/me에 먼저 매칭
mux.HandleFunc("GET /users/{id}", getUser)         // /users/123에 매칭

// 메서드 특정이 범용보다 우선
mux.HandleFunc("GET /items", listItems)            // GET만
mux.HandleFunc("/items", handleItems)              // 다른 모든 메서드
```

---

## 3. Handler와 HandlerFunc

### 3.1 Handler 인터페이스

```go
// http.Handler 인터페이스
type Handler interface {
    ServeHTTP(ResponseWriter, *Request)
}

// 구조체 기반 핸들러
type APIHandler struct {
    db     *sql.DB
    logger *slog.Logger
}

func (h *APIHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    h.logger.Info("request",
        "method", r.Method,
        "path", r.URL.Path,
    )
    // 요청 처리...
}

func main() {
    handler := &APIHandler{
        db:     connectDB(),
        logger: slog.Default(),
    }
    http.Handle("/api/", handler)
}
```

### 3.2 요청 읽기

```go
func handleRequest(w http.ResponseWriter, r *http.Request) {
    // 메서드
    fmt.Println("Method:", r.Method)

    // URL 구성 요소
    fmt.Println("Path:", r.URL.Path)
    fmt.Println("Query:", r.URL.Query().Get("page"))

    // 헤더
    fmt.Println("Content-Type:", r.Header.Get("Content-Type"))
    fmt.Println("User-Agent:", r.Header.Get("User-Agent"))

    // 바디 (POST/PUT용)
    body, err := io.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Bad request", http.StatusBadRequest)
        return
    }
    defer r.Body.Close()
    fmt.Println("Body:", string(body))

    // 폼 데이터
    r.ParseForm()
    fmt.Println("Name:", r.FormValue("name"))

    // JSON 바디
    var data struct {
        Name  string `json:"name"`
        Email string `json:"email"`
    }
    if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
}
```

### 3.3 응답 쓰기

```go
func jsonResponse(w http.ResponseWriter, status int, data any) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(data)
}

func handleUser(w http.ResponseWriter, r *http.Request) {
    user := User{ID: 1, Name: "Alice"}
    jsonResponse(w, http.StatusOK, user)
}

func handleError(w http.ResponseWriter, r *http.Request) {
    // http.Error — 간단한 텍스트 에러 응답
    http.Error(w, "Not Found", http.StatusNotFound)

    // JSON 에러
    jsonResponse(w, http.StatusBadRequest, map[string]string{
        "error": "invalid request",
    })
}
```

---

## 4. 미들웨어

### 4.1 미들웨어 패턴

```go
type Middleware func(http.Handler) http.Handler

// 로깅 미들웨어
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        slog.Info("request",
            "method", r.Method,
            "path", r.URL.Path,
            "duration", time.Since(start),
        )
    })
}

// 복구 미들웨어
func recoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                slog.Error("panic recovered", "error", err)
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// CORS 미들웨어
func corsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        next.ServeHTTP(w, r)
    })
}
```

### 4.2 미들웨어 체이닝

```go
func chain(handler http.Handler, middlewares ...Middleware) http.Handler {
    for i := len(middlewares) - 1; i >= 0; i-- {
        handler = middlewares[i](handler)
    }
    return handler
}

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("GET /api/users", listUsers)

    // 미들웨어 체인 적용
    handler := chain(mux,
        recoveryMiddleware,
        loggingMiddleware,
        corsMiddleware,
    )

    http.ListenAndServe(":8080", handler)
}
```

### 4.3 상태 캡처를 위한 응답 래퍼

```go
type statusRecorder struct {
    http.ResponseWriter
    statusCode int
}

func (r *statusRecorder) WriteHeader(code int) {
    r.statusCode = code
    r.ResponseWriter.WriteHeader(code)
}

func loggingWithStatus(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        recorder := &statusRecorder{ResponseWriter: w, statusCode: 200}
        start := time.Now()
        next.ServeHTTP(recorder, r)
        slog.Info("request",
            "method", r.Method,
            "path", r.URL.Path,
            "status", recorder.statusCode,
            "duration", time.Since(start),
        )
    })
}
```

---

## 5. 템플릿과 정적 파일

### 5.1 HTML 템플릿

```go
import "html/template"

var templates = template.Must(template.ParseGlob("templates/*.html"))

type PageData struct {
    Title string
    Users []User
}

func homeHandler(w http.ResponseWriter, r *http.Request) {
    data := PageData{
        Title: "User List",
        Users: []User{
            {ID: 1, Name: "Alice"},
            {ID: 2, Name: "Bob"},
        },
    }
    templates.ExecuteTemplate(w, "home.html", data)
}
```

```html
<!-- templates/home.html -->
<!DOCTYPE html>
<html>
<head><title>{{.Title}}</title></head>
<body>
    <h1>{{.Title}}</h1>
    <ul>
    {{range .Users}}
        <li>{{.Name}} (ID: {{.ID}})</li>
    {{end}}
    </ul>
</body>
</html>
```

### 5.2 정적 파일

```go
func main() {
    mux := http.NewServeMux()

    // 정적 파일 제공
    fs := http.FileServer(http.Dir("static"))
    mux.Handle("GET /static/", http.StripPrefix("/static/", fs))

    // 임베디드 파일 (Go 1.16+)
    //go:embed static/*
    // var staticFS embed.FS
    // mux.Handle("/static/", http.FileServer(http.FS(staticFS)))

    http.ListenAndServe(":8080", mux)
}
```

---

## 6. 우아한 종료

```go
func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("GET /", homeHandler)

    server := &http.Server{
        Addr:    ":8080",
        Handler: mux,
    }

    // 고루틴에서 서버 시작
    go func() {
        slog.Info("server starting", "addr", server.Addr)
        if err := server.ListenAndServe(); err != http.ErrServerClosed {
            slog.Error("server error", "err", err)
        }
    }()

    // 인터럽트 신호 대기
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit

    slog.Info("shutting down server...")

    // 활성 연결에 30초 여유 부여
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if err := server.Shutdown(ctx); err != nil {
        slog.Error("forced shutdown", "err", err)
    }

    slog.Info("server stopped")
}
```

---

## 7. 요약

### 핵심 포인트

1. **`net/http`는 프로덕션 준비 완료이다** — 대부분의 애플리케이션에 프레임워크가 필요 없다.
2. **Go 1.22 향상된 라우팅** — 표준 라이브러리에서 메서드 기반 라우팅과 경로 매개변수를 지원한다.
3. **미들웨어 패턴** — 횡단 관심사를 위한 `func(http.Handler) http.Handler`이다.
4. **항상 타임아웃을 설정한다** — `ReadTimeout`, `WriteTimeout`, `IdleTimeout`이 리소스 고갈을 방지한다.
5. **우아한 종료** — `server.Shutdown(ctx)`가 활성 연결이 완료될 때까지 기다린다.
6. **응답 헤더는 바디 전에** — 바디를 쓰기 전에 `w.Header().Set()`과 `w.WriteHeader()`를 호출한다.

---

## 연습 문제

### 연습 1: 파일 업로드 서버
POST로 파일 업로드를 받아 디렉토리에 저장하고, GET으로 다시 제공하는 서버를 구축한다.

### 연습 2: 미들웨어 모음
요청 ID 주입, 속도 제한, 기본 인증, 요청/응답 로깅이 포함된 미들웨어 패키지를 생성한다.

### 연습 3: SSE 서버
연결된 클라이언트에 실시간 업데이트를 스트리밍하는 Server-Sent Events 엔드포인트를 구현한다.

### 연습 4: 리버스 프록시
커스텀 헤더 주입과 로깅이 포함된 `httputil.ReverseProxy`를 사용하여 간단한 리버스 프록시를 구축한다.
