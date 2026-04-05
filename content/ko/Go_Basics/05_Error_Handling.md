# 05. 에러 처리

**이전**: [인터페이스](./04_Interfaces.md) | **다음**: [패키지와 모듈](./06_Packages_and_Modules.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. `error` 인터페이스를 사용하고 커스텀 에러 타입을 생성할 수 있다
2. `fmt.Errorf`와 `%w`로 에러를 래핑(wrap)하고 언래핑(unwrap)할 수 있다
3. 에러 검사를 위해 `errors.Is`와 `errors.As`를 사용할 수 있다
4. 센티넬 에러(sentinel error)와 에러 타입 계층을 정의할 수 있다
5. `panic`과 `recover`를 적절하게 적용할 수 있다

---

Go는 예외(exception) 대신 명시적 에러 값을 선호하는 것으로 유명하다. 실패할 수 있는 모든 함수는 마지막 반환값으로 `error`를 반환한다. 이 접근 방식은 장황하지만 에러 경로를 가시적으로 만들고 개발자가 매 단계마다 실패에 대해 생각하도록 강제한다.

## 목차
1. [error 인터페이스](#1-error-인터페이스)
2. [에러 생성](#2-에러-생성)
3. [에러 래핑](#3-에러-래핑)
4. [센티넬 에러](#4-센티넬-에러)
5. [커스텀 에러 타입](#5-커스텀-에러-타입)
6. [panic과 recover](#6-panic과-recover)
7. [요약](#7-요약)

---

## 1. error 인터페이스

### 1.1 에러 기초

```go
package main

import (
    "errors"
    "fmt"
    "os"
    "strconv"
)

func main() {
    // 대부분의 함수는 (결과, error)를 반환한다
    f, err := os.Open("nonexistent.txt")
    if err != nil {
        fmt.Println("에러:", err)
        // f를 사용하지 마라 — err != nil이면 유효하지 않다
    } else {
        defer f.Close()
        fmt.Println("열림:", f.Name())
    }

    // 항상 에러를 확인하라 — 이유가 없으면 _를 사용하지 마라
    n, err := strconv.Atoi("not-a-number")
    if err != nil {
        fmt.Println("파싱 에러:", err)
        return
    }
    fmt.Println("파싱됨:", n)
}
```

### 1.2 에러 패턴

```go
// 패턴 1: 에러 시 조기 반환 (가드 절)
func readConfig(path string) (Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return Config{}, err
    }

    var cfg Config
    err = json.Unmarshal(data, &cfg)
    if err != nil {
        return Config{}, err
    }

    return cfg, nil
}

// 패턴 2: 연속적인 여러 에러 확인
func processFile(path string) error {
    f, err := os.Open(path)
    if err != nil {
        return fmt.Errorf("open: %w", err)
    }
    defer f.Close()

    data, err := io.ReadAll(f)
    if err != nil {
        return fmt.Errorf("read: %w", err)
    }

    if err := validate(data); err != nil {
        return fmt.Errorf("validate: %w", err)
    }

    return nil
}

// 패턴 3: defer와 에러 변수
func writeFile(path string, data []byte) (err error) {
    f, err := os.Create(path)
    if err != nil {
        return err
    }
    defer func() {
        closeErr := f.Close()
        if err == nil {
            err = closeErr
        }
    }()

    _, err = f.Write(data)
    return err
}
```

---

## 2. 에러 생성

### 2.1 간단한 에러

```go
import (
    "errors"
    "fmt"
)

func divide(a, b float64) (float64, error) {
    if b == 0 {
        // errors.New — 간단한 정적 에러 메시지
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

func sqrt(x float64) (float64, error) {
    if x < 0 {
        // fmt.Errorf — 형식화된 에러 메시지
        return 0, fmt.Errorf("cannot take sqrt of negative number: %g", x)
    }
    return math.Sqrt(x), nil
}

func main() {
    result, err := divide(10, 0)
    if err != nil {
        fmt.Println(err) // "division by zero"
    }

    result, err = sqrt(-1)
    if err != nil {
        fmt.Println(err) // "cannot take sqrt of negative number: -1"
    }
    _ = result
}
```

### 2.2 에러 메시지 규칙

```go
// 좋음: 소문자, 구두점 없음, 컨텍스트 포함
return fmt.Errorf("open config: %w", err)
return fmt.Errorf("parse port %q: %w", portStr, err)
return errors.New("missing required field: email")

// 나쁨: 대문자, 구두점, 모호함
return fmt.Errorf("Error opening file.")        // 나쁨
return errors.New("Something went wrong!")       // 나쁨
return fmt.Errorf("Failed to process request")   // 나쁨

// 에러 메시지는 체인을 형성한다: "open config: parse port "abc": strconv.Atoi: invalid syntax"
```

---

## 3. 에러 래핑

### 3.1 %w로 래핑하기

Go 1.13에서 `fmt.Errorf`의 `%w`를 사용한 에러 래핑이 도입되었다.

```go
func readFile(path string) ([]byte, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("readFile(%s): %w", path, err)
    }
    return data, nil
}

func loadConfig(path string) (*Config, error) {
    data, err := readFile(path)
    if err != nil {
        return nil, fmt.Errorf("loadConfig: %w", err)
    }

    var cfg Config
    if err := json.Unmarshal(data, &cfg); err != nil {
        return nil, fmt.Errorf("loadConfig: parse json: %w", err)
    }
    return &cfg, nil
}

func main() {
    cfg, err := loadConfig("/nonexistent/config.json")
    if err != nil {
        fmt.Println(err)
        // "loadConfig: readFile(/nonexistent/config.json): open /nonexistent/config.json: no such file or directory"
    }
    _ = cfg
}
```

### 3.2 errors.Is — 에러 동일성 확인

```go
func main() {
    _, err := os.Open("/nonexistent")

    // errors.Is는 래핑 체인을 순회한다
    if errors.Is(err, os.ErrNotExist) {
        fmt.Println("파일을 찾을 수 없다!")
    }

    // 여러 겹의 래핑을 통해서도 동작한다
    wrapped := fmt.Errorf("config: %w",
        fmt.Errorf("read: %w", os.ErrNotExist))
    fmt.Println(errors.Is(wrapped, os.ErrNotExist)) // true
}
```

### 3.3 errors.As — 에러 타입 추출

```go
type PathError struct {
    Op   string
    Path string
    Err  error
}

func (e *PathError) Error() string {
    return fmt.Sprintf("%s %s: %s", e.Op, e.Path, e.Err)
}

func (e *PathError) Unwrap() error {
    return e.Err
}

func main() {
    _, err := os.Open("/nonexistent")

    // 특정 에러 타입을 추출한다
    var pathErr *os.PathError
    if errors.As(err, &pathErr) {
        fmt.Println("연산:", pathErr.Op)
        fmt.Println("경로:", pathErr.Path)
        fmt.Println("기본 에러:", pathErr.Err)
    }
}
```

### 3.4 다중 래핑 (Go 1.20+)

```go
// 여러 에러를 결합한다
func validateForm(name, email string) error {
    var errs []error

    if name == "" {
        errs = append(errs, errors.New("name is required"))
    }
    if email == "" {
        errs = append(errs, errors.New("email is required"))
    }
    if !strings.Contains(email, "@") {
        errs = append(errs, errors.New("email must contain @"))
    }

    if len(errs) > 0 {
        return errors.Join(errs...)
    }
    return nil
}

func main() {
    err := validateForm("", "invalid")
    if err != nil {
        fmt.Println(err)
        // "name is required\nemail must contain @"
    }
}
```

---

## 4. 센티넬 에러

### 4.1 센티넬 에러 정의

센티넬 에러(sentinel error)는 비교를 위해 사용되는 미리 정의된 에러 값이다.

```go
package mypackage

import "errors"

// 센티넬 에러 — 패키지 수준 변수
var (
    ErrNotFound     = errors.New("not found")
    ErrUnauthorized = errors.New("unauthorized")
    ErrConflict     = errors.New("conflict: resource already exists")
    ErrInternal     = errors.New("internal server error")
)

type UserStore struct {
    users map[string]*User
}

func (s *UserStore) Get(id string) (*User, error) {
    u, ok := s.users[id]
    if !ok {
        return nil, fmt.Errorf("user %s: %w", id, ErrNotFound)
    }
    return u, nil
}

// 호출자는 errors.Is로 확인한다
func main() {
    store := &UserStore{users: make(map[string]*User)}

    _, err := store.Get("abc123")
    if errors.Is(err, ErrNotFound) {
        fmt.Println("사용자를 찾을 수 없다, 생성 중...")
    }

    switch {
    case errors.Is(err, ErrNotFound):
        // 404 처리
    case errors.Is(err, ErrUnauthorized):
        // 401 처리
    case err != nil:
        // 예상치 못한 에러 처리
    }
}
```

### 4.2 표준 라이브러리 센티넬

```go
import (
    "io"
    "os"
    "database/sql"
)

// 표준 라이브러리의 일반적인 센티넬 에러:
// io.EOF            — 입력의 끝
// os.ErrNotExist    — 파일을 찾을 수 없음
// os.ErrPermission  — 권한 거부
// sql.ErrNoRows     — 쿼리가 행을 반환하지 않음

func readUntilEOF(r io.Reader) error {
    buf := make([]byte, 1024)
    for {
        _, err := r.Read(buf)
        if errors.Is(err, io.EOF) {
            return nil // 정상적인 입력의 끝
        }
        if err != nil {
            return err // 실제 에러
        }
    }
}
```

---

## 5. 커스텀 에러 타입

### 5.1 구조체 기반 에러

```go
type HTTPError struct {
    Code    int
    Message string
    Details map[string]string
}

func (e *HTTPError) Error() string {
    return fmt.Sprintf("HTTP %d: %s", e.Code, e.Message)
}

func NewHTTPError(code int, msg string) *HTTPError {
    return &HTTPError{Code: code, Message: msg}
}

func fetchUser(id string) (*User, error) {
    if id == "" {
        return nil, &HTTPError{
            Code:    400,
            Message: "user ID is required",
            Details: map[string]string{"field": "id"},
        }
    }
    // ... 데이터베이스에서 가져온다
    return nil, &HTTPError{Code: 404, Message: "user not found"}
}

func main() {
    _, err := fetchUser("")
    if err != nil {
        var httpErr *HTTPError
        if errors.As(err, &httpErr) {
            fmt.Printf("상태 %d: %s\n", httpErr.Code, httpErr.Message)
            fmt.Println("상세:", httpErr.Details)
        }
    }
}
```

### 5.2 Unwrap을 가진 에러 타입

```go
type QueryError struct {
    Query string
    Err   error
}

func (e *QueryError) Error() string {
    return fmt.Sprintf("query %q: %v", e.Query, e.Err)
}

func (e *QueryError) Unwrap() error {
    return e.Err
}

func runQuery(q string) error {
    // 연결 에러를 시뮬레이션한다
    return &QueryError{
        Query: q,
        Err:   fmt.Errorf("connection refused"),
    }
}
```

### 5.3 에러 처리 전략

```go
// 계층 1: 저수준 — 래핑된 에러를 반환한다
func readFromDB(id string) ([]byte, error) {
    row := db.QueryRow("SELECT data FROM items WHERE id = ?", id)
    var data []byte
    if err := row.Scan(&data); err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, fmt.Errorf("item %s: %w", id, ErrNotFound)
        }
        return nil, fmt.Errorf("query item %s: %w", id, err)
    }
    return data, nil
}

// 계층 2: 비즈니스 로직 — 컨텍스트를 추가한다
func GetItem(id string) (*Item, error) {
    data, err := readFromDB(id)
    if err != nil {
        return nil, fmt.Errorf("GetItem: %w", err)
    }
    return parseItem(data)
}

// 계층 3: HTTP 핸들러 — 상태 코드로 매핑한다
func handleGetItem(w http.ResponseWriter, r *http.Request) {
    item, err := GetItem(r.URL.Query().Get("id"))
    if err != nil {
        switch {
        case errors.Is(err, ErrNotFound):
            http.Error(w, "Not Found", 404)
        case errors.Is(err, ErrUnauthorized):
            http.Error(w, "Unauthorized", 401)
        default:
            log.Printf("예상치 못한 에러: %v", err)
            http.Error(w, "Internal Server Error", 500)
        }
        return
    }
    json.NewEncoder(w).Encode(item)
}
```

---

## 6. panic과 recover

### 6.1 panic

`panic`은 복구 불가능한 에러를 위한 것이다 — 프로그래밍 버그이지, 운영상의 에러가 아니다.

```go
func main() {
    // panic의 적절한 사용:
    // 1. 진정으로 불가능한 상태 (프로그래밍 버그)
    // 2. init()이나 main()에서의 초기화 실패

    // 예: 불가능한 상태
    switch dayOfWeek {
    case 0: // 일요일
    case 1: // 월요일
    // ...
    default:
        panic(fmt.Sprintf("invalid day: %d", dayOfWeek))
    }

    // 예: 반드시 성공해야 하는 초기화
    re := regexp.MustCompile(`\d+`) // 정규식이 유효하지 않으면 패닉
    template.Must(template.New("t").Parse("{{.Name}}")) // 파싱 에러 시 패닉
}

// panic을 사용하지 말아야 하는 경우:
// - 파일을 찾을 수 없음 → 에러를 반환한다
// - 네트워크 타임아웃 → 에러를 반환한다
// - 잘못된 사용자 입력 → 에러를 반환한다
```

### 6.2 recover

```go
func safeDiv(a, b int) (result int, err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()
    return a / b, nil // b == 0이면 패닉
}

func main() {
    result, err := safeDiv(10, 0)
    if err != nil {
        fmt.Println("복구됨:", err)
    } else {
        fmt.Println("결과:", result)
    }
}
```

### 6.3 서버에서의 recover

```go
// 하나의 패닉이 전체 서버를 중단시키는 것을 방지하는 미들웨어
func recoveryMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("PANIC: %v\n%s", err, debug.Stack())
                http.Error(w, "Internal Server Error", 500)
            }
        }()
        next.ServeHTTP(w, r)
    })
}
```

---

## 7. 요약

### 핵심 포인트

1. **에러는 값이다** — `error` 인터페이스는 하나의 메서드를 가진다: `Error() string`. 에러를 명시적으로 반환한다.
2. **모든 에러를 확인하라** — `if err != nil`은 Go에서 가장 일반적인 패턴이다. 에러를 조용히 무시하지 마라.
3. **컨텍스트와 함께 래핑하라** — `fmt.Errorf("operation: %w", err)`는 체인을 보존하고 의미를 추가한다.
4. **센티넬에는 `errors.Is`** — 래핑 체인을 통해 비교한다: `errors.Is(err, ErrNotFound)`.
5. **타입에는 `errors.As`** — 타입이 있는 에러를 추출한다: `errors.As(err, &httpErr)`.
6. **예상된 실패에는 센티넬 에러** — `ErrNotFound`, `io.EOF`, `sql.ErrNoRows`.
7. **버그에만 panic** — 불가능한 상태에는 `panic`을 사용하고, 운영상의 에러에는 절대 사용하지 마라.

### 에러 결정 트리

```
이것은 프로그래밍 버그인가?
├── 예 → panic (nil 맵 접근, 불가능한 열거값)
└── 아니오 → error를 반환한다
     │
     이것은 예상된 실패인가?
     ├── 예 → 센티넬 에러 (ErrNotFound, ErrTimeout)
     └── 아니오 → 컨텍스트와 래핑: fmt.Errorf("op: %w", err)
```

---

## 연습 문제

### 연습 1: 검증 라이브러리
여러 필드 에러를 수집하는 `ValidationError` 타입을 만들라. `Error()`, `Unwrap()`, `HasField(name string) bool` 메서드를 구현하라.

### 연습 2: 파일 처리기
각 단계가 컨텍스트와 함께 에러를 래핑하는 파일 처리 함수 체인(열기, 읽기, 파싱, 검증)을 작성하라. `errors.Is`와 `errors.As`로 테스트하라.

### 연습 3: 에러와 재시도
실패 시 재시도하고 모든 시도 정보와 함께 마지막 에러를 반환하는 `Retry(attempts int, delay time.Duration, fn func() error) error` 함수를 구현하라.

### 연습 4: 에러 미들웨어
커스텀 에러 타입을 적절한 HTTP 상태 코드와 JSON 에러 응답으로 변환하는 HTTP 에러 처리 미들웨어를 만들라.
