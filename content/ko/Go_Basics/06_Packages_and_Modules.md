# 06. 패키지와 모듈

**이전**: [에러 처리](./05_Error_Handling.md) | **다음**: [동시성: 고루틴](./07_Concurrency_Goroutines.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 표준 규칙을 따르는 Go 패키지를 생성하고 구성할 수 있다
2. Go 모듈(module)을 사용하여 의존성을 관리할 수 있다
3. 가시성 규칙(내보내기 vs 내보내지 않기)을 이해할 수 있다
4. 깔끔한 API를 위한 패키지 설계 원칙을 적용할 수 있다
5. `go mod` 명령어를 사용하여 의존성 관리와 버전 관리를 할 수 있다

---

Go의 모듈 시스템은 재현 가능한 빌드와 명시적 의존성 관리를 제공한다. 패키지(package)는 코드를 논리적 단위로 구성하고, 모듈(module)은 관련 패키지를 그룹화하고 버전을 관리한다.

## 목차
1. [패키지 기초](#1-패키지-기초)
2. [가시성과 네이밍](#2-가시성과-네이밍)
3. [Go 모듈](#3-go-모듈)
4. [의존성 관리](#4-의존성-관리)
5. [패키지 설계 패턴](#5-패키지-설계-패턴)
6. [internal 패키지와 워크스페이스](#6-internal-패키지와-워크스페이스)
7. [요약](#7-요약)

---

## 1. 패키지 기초

### 1.1 패키지 선언

모든 Go 파일은 패키지 선언으로 시작한다. 같은 디렉토리의 파일은 같은 패키지 이름을 사용해야 한다.

```go
// file: mathutil/math.go
package mathutil

func Add(a, b int) int { return a + b }
func Sub(a, b int) int { return a - b }

// 내보내지 않는 헬퍼 — 이 패키지 내에서만 보인다
func abs(n int) int {
    if n < 0 {
        return -n
    }
    return n
}
```

```go
// file: mathutil/stats.go
package mathutil // 같은 패키지 — 같은 디렉토리

func Mean(nums []float64) float64 {
    if len(nums) == 0 {
        return 0
    }
    sum := 0.0
    for _, n := range nums {
        sum += n
    }
    return sum / float64(len(nums))
}
```

### 1.2 패키지 가져오기

```go
package main

import (
    "fmt"
    "math/rand"
    "strings"

    // 서드파티 패키지
    "github.com/gorilla/mux"

    // 로컬 패키지
    "github.com/username/myproject/mathutil"
    "github.com/username/myproject/internal/config"
)

func main() {
    fmt.Println(mathutil.Add(3, 4))
    fmt.Println(mathutil.Mean([]float64{1, 2, 3, 4, 5}))
}
```

### 1.3 import 그룹화 규칙

```go
import (
    // 그룹 1: 표준 라이브러리
    "fmt"
    "net/http"
    "os"

    // 그룹 2: 서드파티 패키지 (빈 줄 구분)
    "github.com/gorilla/mux"
    "go.uber.org/zap"

    // 그룹 3: 내부/로컬 패키지 (빈 줄 구분)
    "github.com/myorg/myproject/internal/config"
    "github.com/myorg/myproject/pkg/auth"
)
```

### 1.4 import 별칭과 특수 import

```go
import (
    "fmt"

    // 별칭 — 이름 충돌을 해결한다
    crand "crypto/rand"
    mrand "math/rand"

    // 점 import — 현재 네임스페이스로 가져온다 (프로덕션에서는 피한다)
    . "math"

    // 빈 import — init()만 실행한다 (부수 효과)
    _ "github.com/lib/pq"           // PostgreSQL 드라이버 등록
    _ "image/png"                     // PNG 디코더 등록
    _ "net/http/pprof"                // pprof HTTP 핸들러 등록
)

func main() {
    fmt.Println(Sqrt(2))            // 점 import를 통한 math.Sqrt
    fmt.Println(mrand.Intn(100))    // math/rand
    // crand.Read(...)              // crypto/rand
}
```

---

## 2. 가시성과 네이밍

### 2.1 내보내기 규칙

```go
package user

// 내보내기(EXPORTED, 대문자 첫 글자) — 다른 패키지에서 보인다
type User struct {
    ID    int     // 내보내는 필드
    Name  string  // 내보내는 필드
    email string  // 내보내지 않는 필드
}

func NewUser(name, email string) *User {  // 내보내는 함수
    return &User{
        ID:    generateID(), // 내부적으로 내보내지 않는 함수를 호출할 수 있다
        Name:  name,
        email: email,
    }
}

func (u *User) Email() string { return u.email }  // 내보내는 메서드 (getter)

// 내보내지 않음 — 이 패키지 내에서만 보인다
func generateID() int {
    // ...
    return 0
}

type validator struct { // 내보내지 않는 타입
    rules []Rule
}
```

### 2.2 패키지 네이밍 규칙

```go
// 좋음: 짧고, 소문자, 단일 단어
package http
package json
package user
package auth
package config

// 나쁨: 장황, 혼합 대소문자, 범용적
package httpHelpers    // 나쁨: camelCase
package common         // 나쁨: 너무 범용적
package utils          // 나쁨: 너무 범용적 — 무슨 유틸?
package base           // 나쁨: 의미 없음

// 패키지 이름은 정규화된 이름의 일부이다
// 좋음:
http.Get(url)          // "HTTP get"으로 읽힌다
json.Marshal(v)        // "JSON marshal"로 읽힌다
auth.NewToken()        // "auth new token"으로 읽힌다

// 나쁨: 반복(stuttering)
http.HTTPGet(url)      // "HTTP HTTP get"
user.UserCreate()      // "user user create"
```

### 2.3 init() 함수

```go
package config

import (
    "log"
    "os"
)

var (
    DatabaseURL string
    Port        int
)

// init은 패키지가 import될 때 자동으로 실행된다
// 파일당 여러 init() 함수가 허용된다 (하지만 권장하지 않는다)
func init() {
    DatabaseURL = os.Getenv("DATABASE_URL")
    if DatabaseURL == "" {
        DatabaseURL = "postgres://localhost:5432/mydb"
    }

    portStr := os.Getenv("PORT")
    if portStr == "" {
        Port = 8080
    } else {
        var err error
        Port, err = strconv.Atoi(portStr)
        if err != nil {
            log.Fatalf("invalid PORT: %s", portStr)
        }
    }
}
```

---

## 3. Go 모듈

### 3.1 모듈 초기화

```bash
# 새 모듈을 생성한다
mkdir myproject && cd myproject
go mod init github.com/username/myproject

# go.mod가 생성된다
cat go.mod
```

```
module github.com/username/myproject

go 1.22
```

### 3.2 go.mod 파일

```
module github.com/username/myproject

go 1.22

require (
    github.com/gorilla/mux v1.8.1
    go.uber.org/zap v1.27.0
    golang.org/x/sync v0.6.0
)

require (
    // 간접 의존성 (자동으로 관리됨)
    go.uber.org/multierr v1.11.0 // indirect
)
```

### 3.3 go.sum 파일

```bash
# go.sum은 검증을 위한 암호화 해시를 포함한다
# 절대 수동으로 편집하지 마라 — go 도구가 관리한다
# 항상 go.sum을 버전 관리에 커밋하라

cat go.sum
# github.com/gorilla/mux v1.8.1 h1:TuMoUvkRETex...
# github.com/gorilla/mux v1.8.1/go.mod h1:DVbg23sW...
```

### 3.4 모듈 명령어

```bash
# 의존성 추가
go get github.com/gorilla/mux@latest
go get github.com/gorilla/mux@v1.8.1     # 특정 버전
go get github.com/gorilla/mux@v1.8       # 최신 패치

# 의존성 업데이트
go get -u ./...                            # 모든 직접 의존성 업데이트
go get -u=patch ./...                      # 패치 업데이트만

# 정리 — 사용하지 않는 것 제거, 누락된 것 추가
go mod tidy

# 벤더 — 의존성을 로컬로 복사
go mod vendor
go build -mod=vendor ./...

# 의존성 다운로드 (CI 캐싱용)
go mod download

# 모듈 그래프 문제 확인
go mod verify

# 의존성 그래프 표시
go mod graph

# 의존성이 필요한 이유 표시
go mod why github.com/some/package

# go.mod를 프로그래밍 방식으로 편집
go mod edit -require github.com/foo/bar@v1.0.0
go mod edit -droprequire github.com/foo/bar
```

---

## 4. 의존성 관리

### 4.1 시맨틱 버전 관리

```
v1.2.3
│ │ └── 패치: 버그 수정, API 변경 없음
│ └──── 마이너: 새 기능, 하위 호환
└────── 메이저: 호환성을 깨는 변경

v0.x.y — 1.0 이전: 안정성 보장 없음
v2.0.0 — import 경로에 메이저 버전: github.com/user/pkg/v2
```

### 4.2 버전 선택

```go
// Go는 최소 버전 선택(Minimum Version Selection, MVS)을 사용한다
// A가 X v1.2.0을 요구하고 B가 X v1.3.0을 요구하면,
// Go는 X v1.3.0을 선택한다 (둘 다 만족하는 최소 버전)

// v2+ 메이저 버전 접미사
import "github.com/user/pkg/v2"        // v2.x.x
import "github.com/user/pkg/v3"        // v3.x.x
// v0과 v1은 접미사가 없다
import "github.com/user/pkg"           // v0.x.x 또는 v1.x.x
```

### 4.3 replace와 exclude 지시자

```
// go.mod

// 의존성을 로컬 복사본으로 대체한다 (개발용)
replace github.com/user/pkg => ../pkg

// 포크로 대체한다
replace github.com/original/pkg => github.com/myfork/pkg v1.2.3

// 버그가 있는 버전을 제외한다
exclude github.com/user/pkg v1.2.0
```

### 4.4 비공개 모듈

```bash
# 비공개 레포지토리용
export GOPRIVATE=github.com/mycompany/*
export GONOSUMDB=github.com/mycompany/*
export GONOPROXY=github.com/mycompany/*

# 또는 go.env에서
go env -w GOPRIVATE=github.com/mycompany/*
```

---

## 5. 패키지 설계 패턴

### 5.1 표준 프로젝트 레이아웃

```
myproject/
├── go.mod
├── go.sum
├── main.go              # 진입점 (package main)
├── cmd/                 # 여러 진입점
│   ├── server/
│   │   └── main.go      # go run ./cmd/server
│   └── cli/
│       └── main.go      # go run ./cmd/cli
├── internal/            # 비공개 패키지 (컴파일러가 강제)
│   ├── config/
│   ├── database/
│   └── middleware/
├── pkg/                 # 공개 라이브러리 패키지
│   ├── auth/
│   └── models/
├── api/                 # API 정의 (protobuf, OpenAPI)
├── web/                 # 정적 에셋, 템플릿
└── scripts/             # 빌드 및 배포 스크립트
```

### 5.2 함수형 옵션 패턴

```go
package server

type Server struct {
    host    string
    port    int
    timeout time.Duration
    logger  *log.Logger
    tls     bool
}

// Option은 Server를 구성하는 함수이다
type Option func(*Server)

func WithPort(port int) Option {
    return func(s *Server) { s.port = port }
}

func WithTimeout(d time.Duration) Option {
    return func(s *Server) { s.timeout = d }
}

func WithLogger(l *log.Logger) Option {
    return func(s *Server) { s.logger = l }
}

func WithTLS(enable bool) Option {
    return func(s *Server) { s.tls = enable }
}

func New(host string, opts ...Option) *Server {
    s := &Server{
        host:    host,
        port:    8080,
        timeout: 30 * time.Second,
        logger:  log.Default(),
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// 사용:
// s := server.New("localhost",
//     server.WithPort(9090),
//     server.WithTimeout(60*time.Second),
//     server.WithTLS(true),
// )
```

### 5.3 패키지 문서화

```go
// Package auth는 myproject 애플리케이션을 위한
// 인증 및 인가 유틸리티를 제공한다.
//
// 기본 사용법:
//
//	token, err := auth.NewToken(userID, auth.WithExpiry(24*time.Hour))
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	claims, err := auth.Validate(token)
package auth

// User는 인증된 사용자를 나타낸다.
// 영값은 유용하지 않다; 인스턴스를 생성하려면 NewUser를 사용하라.
type User struct {
    ID    string
    Email string
    Roles []string
}
```

---

## 6. internal 패키지와 워크스페이스

### 6.1 internal 패키지

```
myproject/
├── internal/
│   └── secret/          # myproject와 하위 패키지에서만 import 가능
│       └── secret.go
├── pkg/
│   └── public/          # 누구나 import 가능
│       └── public.go
└── cmd/
    └── app/
        └── main.go      # internal/secret을 import할 수 있다
```

```go
// myproject 내에서는 이 import가 동작한다:
import "github.com/user/myproject/internal/secret"

// myproject 외부에서는 이 import가 실패한다:
// import "github.com/user/myproject/internal/secret"
// 에러: use of internal package not allowed
```

### 6.2 Go 워크스페이스 (Go 1.18+)

```bash
# 멀티 모듈 개발용
mkdir workspace && cd workspace
go work init ./module-a ./module-b

# go.work 파일:
cat go.work
```

```
go 1.22

use (
    ./module-a
    ./module-b
)
```

```bash
# 모듈은 replace 지시자 없이 서로를 참조할 수 있다
go work sync  # 모듈 간 go.sum 파일 동기화
```

### 6.3 빌드 태그

```go
//go:build linux
// +build linux

package mypackage

// 이 파일은 Linux에서만 컴파일된다
func platformSpecific() {
    // Linux 전용 구현
}

// file: mypackage_windows.go
//go:build windows

func platformSpecific() {
    // Windows 전용 구현
}
```

```bash
# 커스텀 태그로 빌드
go build -tags integration ./...
```

---

## 7. 요약

### 핵심 포인트

1. **패키지는 코드를 구성한다** — 하나의 디렉토리 = 하나의 패키지. 파일 이름은 중요하지 않고, 패키지 선언만 중요하다.
2. **대문자 = 내보내기** — 유일한 가시성 규칙이다. public/private/protected 키워드가 없다.
3. **Go 모듈이 의존성을 관리한다** — `go.mod`가 모듈 경로와 요구 사항을 선언한다.
4. **`go mod tidy`는 친구이다** — import를 추가/제거한 후 실행하여 `go.mod`를 깔끔하게 유지한다.
5. **internal 패키지는 강제된다** — `internal/` 디렉토리는 import 가능성을 제한한다.
6. **작고 집중된 패키지** — `util`, `common`, `helper`를 피한다. 패키지 이름이 목적을 설명해야 한다.
7. **구성을 위한 함수형 옵션** — `WithXxx` 패턴은 매개변수 폭발을 방지한다.

### 빠른 참조

```bash
go mod init MODULE_PATH    # 모듈 초기화
go mod tidy                # 의존성 동기화
go get PKG@VERSION         # 의존성 추가/업데이트
go mod vendor              # 의존성 벤더링
go build ./...             # 모든 패키지 빌드
go test ./...              # 모든 패키지 테스트
go vet ./...               # 정적 분석
go doc PKG                 # 문서 보기
```

---

## 연습 문제

### 연습 1: 패키지 구성
세 개의 패키지를 가진 프로젝트를 만들라: `models` (User, Product 구조체), `store` (인메모리 CRUD), `main` (CLI 인터페이스). 적절한 가시성과 네이밍을 연습하라.

### 연습 2: 함수형 옵션
함수형 옵션을 가진 `Logger`를 구현하라: `WithLevel`, `WithOutput`, `WithFormat`, `WithTimestamp`. 깔끔한 패키지 문서를 작성하라.

### 연습 3: 멀티 모듈 워크스페이스
두 개의 모듈을 가진 워크스페이스를 만들라: `mathlib` (수학 유틸리티)와 `calculator` (mathlib를 사용하는 CLI). 로컬 개발에 `go work`를 사용하라.

### 연습 4: 빌드 태그
두 개의 구현을 가진 `storage` 패키지를 만들라: `storage_file.go` (파일 기반)와 `storage_memory.go` (메모리 기반), 빌드 태그로 선택한다.
