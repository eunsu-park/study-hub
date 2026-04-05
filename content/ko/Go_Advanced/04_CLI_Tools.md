# 15. CLI 도구

**이전**: [데이터베이스 접근](./03_Database_Access.md) | **다음**: [고급 타입](./05_Advanced_Types.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. `flag` 패키지로 커맨드라인 플래그를 파싱한다
2. `cobra`로 다중 명령 CLI를 구축한다
3. Unix 스타일 파이핑을 위한 stdin/stdout을 처리한다
4. 대화형 터미널 애플리케이션을 만든다
5. 진행률 표시기와 색상 출력을 구현한다

---

Go는 CLI 도구를 위한 최고의 언어이다 — Docker, Kubernetes(kubectl), Terraform, Hugo 모두 Go CLI 애플리케이션이다. Go의 빠른 컴파일, 정적 바이너리, 뛰어난 크로스 컴파일 기능이 이를 가능하게 한다.

## 목차
1. [flag 패키지](#1-flag-패키지)
2. [Cobra 프레임워크](#2-cobra-프레임워크)
3. [stdin/stdout과 파이핑](#3-stdinstdout과-파이핑)
4. [환경 변수와 설정](#4-환경-변수와-설정)
5. [터미널 UI](#5-터미널-ui)
6. [배포](#6-배포)
7. [요약](#7-요약)

---

## 1. flag 패키지

### 1.1 기본 플래그

```go
package main

import (
    "flag"
    "fmt"
)

func main() {
    // Define flags
    name := flag.String("name", "World", "name to greet")
    count := flag.Int("count", 1, "number of greetings")
    verbose := flag.Bool("verbose", false, "enable verbose output")
    port := flag.Int("port", 8080, "server port")

    // Parse command-line arguments
    flag.Parse()

    // Use flags
    for i := 0; i < *count; i++ {
        fmt.Printf("Hello, %s!\n", *name)
    }

    if *verbose {
        fmt.Printf("Port: %d\n", *port)
    }

    // Remaining arguments (not flags)
    args := flag.Args()
    fmt.Println("Extra args:", args)
}
```

```bash
go run main.go -name Alice -count 3 -verbose extra1 extra2
```

### 1.2 FlagSet을 사용한 서브커맨드

```go
func main() {
    // Subcommand flag sets
    serveCmd := flag.NewFlagSet("serve", flag.ExitOnError)
    servePort := serveCmd.Int("port", 8080, "server port")
    serveHost := serveCmd.String("host", "localhost", "server host")

    initCmd := flag.NewFlagSet("init", flag.ExitOnError)
    initTemplate := initCmd.String("template", "default", "project template")

    if len(os.Args) < 2 {
        fmt.Println("Usage: app <command> [flags]")
        fmt.Println("Commands: serve, init")
        os.Exit(1)
    }

    switch os.Args[1] {
    case "serve":
        serveCmd.Parse(os.Args[2:])
        fmt.Printf("Serving on %s:%d\n", *serveHost, *servePort)
    case "init":
        initCmd.Parse(os.Args[2:])
        fmt.Printf("Initializing with template: %s\n", *initTemplate)
    default:
        fmt.Printf("Unknown command: %s\n", os.Args[1])
        os.Exit(1)
    }
}
```

---

## 2. Cobra 프레임워크

### 2.1 프로젝트 구조

```
myapp/
├── cmd/
│   ├── root.go
│   ├── serve.go
│   ├── init.go
│   └── version.go
├── internal/
│   └── ...
├── main.go
└── go.mod
```

### 2.2 루트 커맨드

```go
// cmd/root.go
package cmd

import (
    "fmt"
    "os"

    "github.com/spf13/cobra"
)

var (
    cfgFile string
    verbose bool
)

var rootCmd = &cobra.Command{
    Use:   "myapp",
    Short: "A brief description of your application",
    Long:  `A longer description that spans multiple lines.`,
}

func Execute() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}

func init() {
    rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file")
    rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")
}
```

### 2.3 서브커맨드

```go
// cmd/serve.go
package cmd

import (
    "fmt"
    "github.com/spf13/cobra"
)

var (
    port int
    host string
)

var serveCmd = &cobra.Command{
    Use:   "serve",
    Short: "Start the HTTP server",
    Long:  `Start the HTTP server with the specified configuration.`,
    RunE: func(cmd *cobra.Command, args []string) error {
        fmt.Printf("Starting server on %s:%d\n", host, port)
        return startServer(host, port)
    },
}

func init() {
    rootCmd.AddCommand(serveCmd)
    serveCmd.Flags().IntVarP(&port, "port", "p", 8080, "server port")
    serveCmd.Flags().StringVar(&host, "host", "localhost", "server host")
}
```

```go
// main.go
package main

import "github.com/username/myapp/cmd"

func main() {
    cmd.Execute()
}
```

---

## 3. stdin/stdout과 파이핑

### 3.1 stdin 읽기

```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"
)

func main() {
    // Check if stdin is a pipe or terminal
    stat, _ := os.Stdin.Stat()
    isPipe := (stat.Mode() & os.ModeCharDevice) == 0

    if isPipe {
        // Reading from pipe: cat file.txt | myapp
        scanner := bufio.NewScanner(os.Stdin)
        lineNum := 0
        for scanner.Scan() {
            lineNum++
            line := scanner.Text()
            fmt.Printf("%4d: %s\n", lineNum, strings.ToUpper(line))
        }
        if err := scanner.Err(); err != nil {
            fmt.Fprintln(os.Stderr, "error reading stdin:", err)
            os.Exit(1)
        }
    } else {
        // Interactive mode
        fmt.Println("Enter text (Ctrl+D to finish):")
        scanner := bufio.NewScanner(os.Stdin)
        for scanner.Scan() {
            fmt.Println("You said:", scanner.Text())
        }
    }
}
```

### 3.2 stdout vs stderr

```go
func main() {
    // Regular output → stdout (piped)
    fmt.Println("This goes to stdout")
    fmt.Fprintln(os.Stdout, "Also stdout")

    // Errors and diagnostics → stderr (not piped)
    fmt.Fprintln(os.Stderr, "This goes to stderr")
    log.Println("Log goes to stderr by default")

    // Usage: myapp 2>/dev/null  — hide errors
    // Usage: myapp > output.txt — redirect stdout only
    // Usage: myapp 2>&1         — merge stderr into stdout
}
```

### 3.3 종료 코드

```go
func main() {
    if err := run(); err != nil {
        fmt.Fprintln(os.Stderr, "error:", err)
        os.Exit(1) // Non-zero = failure
    }
    // os.Exit(0) is implicit on normal return
}

func run() error {
    // Actual application logic
    return nil
}

// Common exit codes:
// 0 — success
// 1 — general error
// 2 — usage error (wrong arguments)
// 126 — permission denied
// 127 — command not found
```

---

## 4. 환경 변수와 설정

### 4.1 환경 변수

```go
func main() {
    // Read environment variables
    dbURL := os.Getenv("DATABASE_URL")
    if dbURL == "" {
        dbURL = "postgres://localhost:5432/myapp"
    }

    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }

    // LookupEnv distinguishes empty from unset
    val, exists := os.LookupEnv("API_KEY")
    if !exists {
        fmt.Println("API_KEY not set")
    } else if val == "" {
        fmt.Println("API_KEY is empty")
    }
}
```

### 4.2 설정 우선순위

```go
type Config struct {
    Host     string
    Port     int
    LogLevel string
    DBUrl    string
}

func LoadConfig() *Config {
    cfg := &Config{
        Host:     "localhost",
        Port:     8080,
        LogLevel: "info",
    }

    // 1. Config file (lowest priority)
    if data, err := os.ReadFile("config.json"); err == nil {
        json.Unmarshal(data, cfg)
    }

    // 2. Environment variables (medium priority)
    if host := os.Getenv("APP_HOST"); host != "" {
        cfg.Host = host
    }
    if port := os.Getenv("APP_PORT"); port != "" {
        cfg.Port, _ = strconv.Atoi(port)
    }

    // 3. Command-line flags (highest priority)
    flag.StringVar(&cfg.Host, "host", cfg.Host, "server host")
    flag.IntVar(&cfg.Port, "port", cfg.Port, "server port")
    flag.Parse()

    return cfg
}
```

---

## 5. 터미널 UI

### 5.1 진행률 표시기

```go
func spinner(done <-chan struct{}) {
    chars := `|/-\`
    i := 0
    for {
        select {
        case <-done:
            fmt.Print("\r \r") // Clear spinner
            return
        default:
            fmt.Printf("\r%c Processing...", chars[i%len(chars)])
            i++
            time.Sleep(100 * time.Millisecond)
        }
    }
}

func main() {
    done := make(chan struct{})
    go spinner(done)

    // Simulate work
    time.Sleep(3 * time.Second)

    close(done)
    fmt.Println("Done!")
}
```

### 5.2 진행률 바

```go
func progressBar(current, total int, width int) string {
    percent := float64(current) / float64(total)
    filled := int(percent * float64(width))
    bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
    return fmt.Sprintf("\r[%s] %3.0f%% (%d/%d)", bar, percent*100, current, total)
}

func main() {
    total := 100
    for i := 0; i <= total; i++ {
        fmt.Print(progressBar(i, total, 40))
        time.Sleep(50 * time.Millisecond)
    }
    fmt.Println()
}
```

### 5.3 ANSI 색상

```go
const (
    Reset  = "\033[0m"
    Red    = "\033[31m"
    Green  = "\033[32m"
    Yellow = "\033[33m"
    Blue   = "\033[34m"
    Bold   = "\033[1m"
)

func success(msg string) {
    fmt.Printf("%s%s✓ %s%s\n", Green, Bold, msg, Reset)
}

func warning(msg string) {
    fmt.Printf("%s⚠ %s%s\n", Yellow, msg, Reset)
}

func errorMsg(msg string) {
    fmt.Printf("%s✗ %s%s\n", Red, msg, Reset)
}
```

---

## 6. 배포

### 6.1 크로스 컴파일

```bash
# Build for Linux
GOOS=linux GOARCH=amd64 go build -o myapp-linux-amd64

# Build for macOS (Apple Silicon)
GOOS=darwin GOARCH=arm64 go build -o myapp-darwin-arm64

# Build for Windows
GOOS=windows GOARCH=amd64 go build -o myapp-windows-amd64.exe

# Common targets
# GOOS: linux, darwin, windows, freebsd
# GOARCH: amd64, arm64, 386, arm
```

### 6.2 빌드 정보

```go
// Set at build time with ldflags
var (
    version = "dev"
    commit  = "unknown"
    date    = "unknown"
)

func main() {
    if os.Args[1] == "version" {
        fmt.Printf("Version: %s\nCommit: %s\nDate: %s\n", version, commit, date)
        return
    }
}
```

```bash
go build -ldflags "-X main.version=1.0.0 -X main.commit=$(git rev-parse HEAD) -X main.date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" -o myapp
```

---

## 7. 요약

### 핵심 포인트

1. **간단한 CLI에는 `flag`** — 내장 패키지로 의존성이 없다. 단일 명령 도구에 적합하다.
2. **복잡한 CLI에는 Cobra** — 서브커맨드, 자동 완성, 도움말 생성을 지원한다.
3. **Unix 관례** — stdout은 데이터용, stderr는 진단용, 종료 코드는 상태용이다.
4. **설정 계층** — 기본값 < 설정 파일 < 환경 변수 < 플래그 순으로 우선한다.
5. **정적 바이너리** — `CGO_ENABLED=0 go build`로 완전히 독립적인 바이너리를 생성한다.
6. **크로스 컴파일** — `GOOS`/`GOARCH` 환경 변수만으로 가능하다. 크로스 컴파일러가 필요 없다.

---

## 연습 문제

### 연습 1: 파일 유틸리티
서브커맨드가 있는 CLI 도구를 만든다: `count`(줄/단어/문자 수), `find`(파일 검색), `replace`(파일 내 찾기 및 바꾸기).

### 연습 2: 작업 관리자
cobra로 작업 관리 CLI를 만든다: `add`, `list`, `done`, `delete` 커맨드와 JSON 파일 영속성을 구현한다.

### 연습 3: Unix 필터
stdin을 읽고 변환(대문자, 소문자, 트림, 줄바꿈)을 적용한 후 stdout으로 출력하는 텍스트 처리 필터를 만든다.

### 연습 4: 릴리스 자동화
3개 플랫폼용 크로스 컴파일을 수행하고, ldflags를 통해 버전 정보를 포함하며, 릴리스 아카이브를 생성하는 빌드 스크립트를 만든다.
