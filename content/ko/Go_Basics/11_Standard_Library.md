# 11. 표준 라이브러리

**이전**: [테스팅](./10_Testing.md) | **다음**: [HTTP 서버](../Go_Advanced/01_HTTP_Server.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 조합 가능한 I/O 연산을 위해 `io` 패키지 인터페이스를 사용한다
2. `encoding/json`으로 JSON을 파싱하고 생성한다
3. `os`와 `filepath`를 사용하여 파일과 디렉토리를 다룬다
4. `net/http`로 HTTP 요청을 만든다
5. `time`, `strings`, `bytes`, `regexp`, `log/slog`를 사용한다

---

Go의 표준 라이브러리는 포괄적인 것으로 유명하다. HTTP 서버부터 암호화까지 서드파티 패키지 없이 모든 것을 포함한다. 이 레슨은 가장 일반적으로 사용되는 패키지를 다룬다.

## 목차
1. [io 패키지](#1-io-패키지)
2. [encoding/json](#2-encodingjson)
3. [os와 filepath](#3-os와-filepath)
4. [net/http 클라이언트](#4-nethttp-클라이언트)
5. [time 패키지](#5-time-패키지)
6. [strings, bytes, regexp](#6-strings-bytes-regexp)
7. [요약](#7-요약)

---

## 1. io 패키지

### 1.1 Reader와 Writer

```go
package main

import (
    "bytes"
    "fmt"
    "io"
    "os"
    "strings"
)

func main() {
    // strings.NewReader → io.Reader
    r := strings.NewReader("Hello, io!")

    // Reader를 stdout(io.Writer)으로 복사
    io.Copy(os.Stdout, r)
    fmt.Println()

    // ReadAll
    r2 := strings.NewReader("read everything")
    data, _ := io.ReadAll(r2)
    fmt.Println(string(data))

    // bytes.Buffer — Reader와 Writer 모두 구현
    var buf bytes.Buffer
    buf.WriteString("hello ")
    buf.WriteString("buffer")
    fmt.Println(buf.String())

    // MultiReader — Reader들을 연결
    r1 := strings.NewReader("part1 ")
    r3 := strings.NewReader("part2")
    multi := io.MultiReader(r1, r3)
    io.Copy(os.Stdout, multi)
    fmt.Println()

    // TeeReader — 읽기와 동시에 복사
    original := strings.NewReader("tee data")
    var captured bytes.Buffer
    tee := io.TeeReader(original, &captured)
    io.Copy(os.Stdout, tee)
    fmt.Println()
    fmt.Println("Captured:", captured.String())

    // LimitReader — N 바이트만 읽기
    big := strings.NewReader("a very long string that we only want part of")
    limited := io.LimitReader(big, 10)
    data, _ = io.ReadAll(limited)
    fmt.Println(string(data)) // "a very lon"
}
```

### 1.2 Pipe

```go
func main() {
    pr, pw := io.Pipe()

    // Writer 고루틴
    go func() {
        defer pw.Close()
        for i := 0; i < 5; i++ {
            fmt.Fprintf(pw, "line %d\n", i)
        }
    }()

    // Reader
    data, _ := io.ReadAll(pr)
    fmt.Println(string(data))
}
```

---

## 2. encoding/json

### 2.1 Marshal과 Unmarshal

```go
package main

import (
    "encoding/json"
    "fmt"
    "time"
)

type User struct {
    ID        int       `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email,omitempty"`
    CreatedAt time.Time `json:"created_at"`
    Password  string    `json:"-"` // 절대 포함되지 않음
}

func main() {
    // Marshal: Go 구조체 → JSON 바이트
    user := User{
        ID:        1,
        Name:      "Alice",
        Email:     "alice@example.com",
        CreatedAt: time.Now(),
        Password:  "secret",
    }

    data, err := json.Marshal(user)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    fmt.Println(string(data))

    // 보기 좋게 출력
    pretty, _ := json.MarshalIndent(user, "", "  ")
    fmt.Println(string(pretty))

    // Unmarshal: JSON 바이트 → Go 구조체
    jsonStr := `{"id": 2, "name": "Bob", "created_at": "2024-01-15T10:30:00Z"}`
    var user2 User
    err = json.Unmarshal([]byte(jsonStr), &user2)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    fmt.Printf("%+v\n", user2)
}
```

### 2.2 스트리밍 JSON

```go
func main() {
    // Encoder — io.Writer에 JSON 쓰기
    encoder := json.NewEncoder(os.Stdout)
    encoder.SetIndent("", "  ")
    encoder.Encode(User{ID: 1, Name: "Alice"})

    // Decoder — io.Reader에서 JSON 읽기
    jsonStream := `{"id": 1, "name": "Alice"}
{"id": 2, "name": "Bob"}`

    decoder := json.NewDecoder(strings.NewReader(jsonStream))
    for decoder.More() {
        var user User
        if err := decoder.Decode(&user); err != nil {
            fmt.Println("Error:", err)
            break
        }
        fmt.Printf("%+v\n", user)
    }
}
```

### 2.3 동적 JSON

```go
func main() {
    // 동적 JSON을 위해 map으로 Unmarshal
    jsonStr := `{"name": "Alice", "age": 30, "scores": [95, 87, 92]}`
    var data map[string]any
    json.Unmarshal([]byte(jsonStr), &data)

    fmt.Println(data["name"])   // "Alice"
    fmt.Println(data["age"])    // 30 (float64!)
    fmt.Println(data["scores"]) // [95 87 92]

    // json.RawMessage — 파싱 지연
    type Event struct {
        Type    string          `json:"type"`
        Payload json.RawMessage `json:"payload"`
    }

    eventJSON := `{"type": "user_created", "payload": {"id": 1, "name": "Alice"}}`
    var event Event
    json.Unmarshal([]byte(eventJSON), &event)

    // 타입에 따라 payload 파싱
    switch event.Type {
    case "user_created":
        var user User
        json.Unmarshal(event.Payload, &user)
        fmt.Printf("Created user: %+v\n", user)
    }
}
```

---

## 3. os와 filepath

### 3.1 파일 연산

```go
package main

import (
    "fmt"
    "os"
)

func main() {
    // 파일 쓰기
    err := os.WriteFile("example.txt", []byte("Hello, file!\n"), 0644)
    if err != nil {
        fmt.Println("Write error:", err)
        return
    }

    // 파일 읽기
    data, err := os.ReadFile("example.txt")
    if err != nil {
        fmt.Println("Read error:", err)
        return
    }
    fmt.Println(string(data))

    // 명시적 open/close를 사용한 파일
    f, err := os.Create("output.txt")
    if err != nil {
        fmt.Println(err)
        return
    }
    defer f.Close()

    f.WriteString("line 1\n")
    f.WriteString("line 2\n")
    fmt.Fprintf(f, "line %d\n", 3)

    // 파일에 추가
    f2, _ := os.OpenFile("output.txt", os.O_APPEND|os.O_WRONLY, 0644)
    defer f2.Close()
    f2.WriteString("line 4 (appended)\n")

    // 존재 여부 확인
    if _, err := os.Stat("example.txt"); os.IsNotExist(err) {
        fmt.Println("File does not exist")
    }

    // 제거
    os.Remove("example.txt")
    os.Remove("output.txt")
}
```

### 3.2 디렉토리 연산

```go
func main() {
    // 디렉토리 생성
    os.Mkdir("testdir", 0755)
    os.MkdirAll("nested/deep/dir", 0755)

    // 디렉토리 항목 읽기
    entries, _ := os.ReadDir(".")
    for _, entry := range entries {
        info, _ := entry.Info()
        fmt.Printf("%-30s %10d %s\n", entry.Name(), info.Size(), info.ModTime().Format("2006-01-02"))
    }

    // 디렉토리 트리 순회
    filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
        if err != nil {
            return err
        }
        fmt.Println(path)
        return nil
    })

    // 정리
    os.RemoveAll("testdir")
    os.RemoveAll("nested")
}
```

### 3.3 filepath 패키지

```go
import "path/filepath"

func main() {
    // 경로 결합 (OS 인식)
    p := filepath.Join("home", "user", "documents", "file.txt")
    fmt.Println(p) // Unix에서 "home/user/documents/file.txt"

    // 구성 요소 추출
    fmt.Println(filepath.Dir(p))   // "home/user/documents"
    fmt.Println(filepath.Base(p))  // "file.txt"
    fmt.Println(filepath.Ext(p))   // ".txt"

    // 경로 정리
    fmt.Println(filepath.Clean("a/b/../c")) // "a/c"

    // 절대 경로
    abs, _ := filepath.Abs(".")
    fmt.Println(abs)

    // Glob 패턴 매칭
    matches, _ := filepath.Glob("*.go")
    fmt.Println(matches)
}
```

---

## 4. net/http 클라이언트

### 4.1 간단한 요청

```go
func main() {
    // GET 요청
    resp, err := http.Get("https://httpbin.org/get")
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)
    fmt.Println("Status:", resp.StatusCode)
    fmt.Println(string(body))

    // JSON을 포함한 POST
    payload := bytes.NewBufferString(`{"name": "Alice"}`)
    resp, err = http.Post("https://httpbin.org/post", "application/json", payload)
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    defer resp.Body.Close()
    body, _ = io.ReadAll(resp.Body)
    fmt.Println(string(body))
}
```

### 4.2 커스텀 클라이언트

```go
func main() {
    client := &http.Client{
        Timeout: 10 * time.Second,
        Transport: &http.Transport{
            MaxIdleConns:        100,
            IdleConnTimeout:     90 * time.Second,
            MaxConnsPerHost:     10,
            MaxIdleConnsPerHost: 10,
        },
    }

    req, err := http.NewRequest("GET", "https://api.example.com/data", nil)
    if err != nil {
        fmt.Println(err)
        return
    }

    req.Header.Set("Authorization", "Bearer token123")
    req.Header.Set("Accept", "application/json")

    resp, err := client.Do(req)
    if err != nil {
        fmt.Println(err)
        return
    }
    defer resp.Body.Close()

    var result map[string]any
    json.NewDecoder(resp.Body).Decode(&result)
    fmt.Println(result)
}
```

---

## 5. time 패키지

### 5.1 시간 연산

```go
func main() {
    now := time.Now()
    fmt.Println("Now:", now)
    fmt.Println("Unix:", now.Unix())

    // 특정 시간 생성
    t := time.Date(2024, time.March, 15, 10, 30, 0, 0, time.UTC)
    fmt.Println(t)

    // Duration
    d := 2*time.Hour + 30*time.Minute
    fmt.Println(d)

    future := now.Add(d)
    fmt.Println("Future:", future)

    elapsed := time.Since(now)
    fmt.Println("Elapsed:", elapsed)

    // 포매팅 — Go는 참조 시간을 사용: Mon Jan 2 15:04:05 MST 2006
    fmt.Println(now.Format("2006-01-02 15:04:05"))
    fmt.Println(now.Format(time.RFC3339))
    fmt.Println(now.Format("January 2, 2006"))

    // 파싱
    parsed, _ := time.Parse("2006-01-02", "2024-03-15")
    fmt.Println(parsed)

    // Timer와 Ticker
    timer := time.NewTimer(2 * time.Second)
    <-timer.C
    fmt.Println("Timer fired")

    ticker := time.NewTicker(500 * time.Millisecond)
    defer ticker.Stop()
    for i := 0; i < 3; i++ {
        <-ticker.C
        fmt.Println("Tick")
    }
}
```

---

## 6. strings, bytes, regexp

### 6.1 strings 패키지

```go
import "strings"

func main() {
    s := "Hello, World!"

    fmt.Println(strings.Contains(s, "World"))     // true
    fmt.Println(strings.HasPrefix(s, "Hello"))     // true
    fmt.Println(strings.HasSuffix(s, "!"))         // true
    fmt.Println(strings.Index(s, "World"))         // 7
    fmt.Println(strings.Count(s, "l"))             // 3
    fmt.Println(strings.Repeat("Go ", 3))          // "Go Go Go "
    fmt.Println(strings.TrimSpace("  hello  "))    // "hello"
    fmt.Println(strings.ReplaceAll(s, "l", "L"))   // "HeLLo, WorLd!"

    // Split과 Join
    parts := strings.Split("a,b,c,d", ",")
    fmt.Println(parts) // [a b c d]
    fmt.Println(strings.Join(parts, " | ")) // "a | b | c | d"

    // strings.Builder
    var b strings.Builder
    for i := 0; i < 1000; i++ {
        fmt.Fprintf(&b, "%d ", i)
    }
    result := b.String()
    _ = result
}
```

### 6.2 regexp 패키지

```go
import "regexp"

func main() {
    // 한 번 컴파일하고 여러 번 재사용
    re := regexp.MustCompile(`\b\w+@\w+\.\w+\b`)

    text := "Contact alice@example.com or bob@test.org"

    // 모든 매치 찾기
    matches := re.FindAllString(text, -1)
    fmt.Println(matches) // [alice@example.com bob@test.org]

    // 치환
    masked := re.ReplaceAllString(text, "[EMAIL]")
    fmt.Println(masked)

    // 서브매치 (캡처 그룹)
    re2 := regexp.MustCompile(`(\w+)@(\w+)\.(\w+)`)
    parts := re2.FindStringSubmatch("alice@example.com")
    fmt.Println(parts) // [alice@example.com alice example com]

    // 이름 있는 그룹
    re3 := regexp.MustCompile(`(?P<user>\w+)@(?P<domain>\w+\.\w+)`)
    match := re3.FindStringSubmatch("alice@example.com")
    for i, name := range re3.SubexpNames() {
        if name != "" {
            fmt.Printf("%s: %s\n", name, match[i])
        }
    }
}
```

### 6.3 log/slog (Go 1.21+)

```go
import "log/slog"

func main() {
    // 구조화된 로깅
    slog.Info("user logged in",
        "user_id", 123,
        "ip", "192.168.1.1",
    )

    // JSON 핸들러
    logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelDebug,
    }))

    logger.Debug("debug message", "key", "value")
    logger.Info("request processed",
        "method", "GET",
        "path", "/api/users",
        "duration_ms", 42,
    )
    logger.Error("database error",
        "err", fmt.Errorf("connection refused"),
        "retry", 3,
    )

    // 기본 필드가 있는 로거
    childLogger := logger.With("service", "auth", "version", "1.0")
    childLogger.Info("token created", "user_id", 42)
}
```

---

## 7. 요약

### 핵심 포인트

1. **io.Reader와 io.Writer** — 조합 가능한 I/O의 기반이다. 대부분의 패키지가 이 인터페이스를 받아들인다.
2. **encoding/json** — 구조체 태그가 마샬링을 제어한다. 지연 파싱에 `json.RawMessage`를 사용한다.
3. **os.ReadFile / os.WriteFile** — 간단한 파일 연산이다. 대용량 파일 스트리밍에는 `os.Open`을 사용한다.
4. **net/http.Client** — 항상 타임아웃을 설정한다. 커넥션 풀링을 위해 클라이언트를 재사용한다.
5. **time 레이아웃은 참조 시간을 사용한다** — `2006-01-02 15:04:05`는 Go만의 고유한 포매팅 방식이다.
6. **strings.Builder** — 효율적인 문자열 연결이다. 루프에서 `+`를 절대 사용하지 않는다.
7. **log/slog** — Go 1.21+에 내장된 구조화된 로깅이다. `log` 패키지 대신 사용한다.

---

## 연습 문제

### 연습 1: JSON 설정 로더
파일에서 JSON을 읽고, 필수 필드를 검증하고, 타입이 지정된 Config 구조체를 반환하는 설정 로더를 작성한다. 환경 변수 오버라이드를 지원한다.

### 연습 2: 파일 워쳐
디렉토리의 변경 사항(새 파일/수정/삭제)을 감시하고 타임스탬프와 함께 이벤트를 출력하는 유틸리티를 생성한다.

### 연습 3: HTTP 클라이언트 래퍼
재시도 로직, 타임아웃, 커스텀 헤더, JSON 응답 디코딩을 갖춘 재사용 가능한 HTTP 클라이언트를 구축한다.

### 연습 4: 로그 분석기
구조화된 로그 파일(JSON 라인)을 읽고, 레벨과 시간 범위로 필터링하고, 요약 보고서를 생성하는 프로그램을 작성한다.
