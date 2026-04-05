# 19. 빌드와 배포

**이전**: [성능 프로파일링](./07_Performance_Profiling.md) | **다음**: [네트워크 프로그래밍](./09_Network_Programming.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 여러 플랫폼용으로 Go 바이너리를 크로스 컴파일한다
2. 멀티 스테이지 빌드로 최소 Docker 이미지를 생성한다
3. 파일을 임베딩하고 `ldflags`로 빌드 시점 변수를 설정한다
4. Go 프로젝트를 위한 CI/CD 파이프라인을 구축한다
5. GoReleaser를 사용하여 자동화된 릴리스 관리를 수행한다

---

Go의 가장 큰 장점 중 하나는 배포의 단순함이다. Go 프로그램은 런타임 의존성이 없는 단일 정적 바이너리로 컴파일된다. 이로 인해 컨테이너화, 크로스 컴파일, 배포가 간단해진다.

## 목차
1. [빌드 기초](#1-빌드-기초)
2. [크로스 컴파일](#2-크로스-컴파일)
3. [Docker](#3-docker)
4. [파일 임베딩](#4-파일-임베딩)
5. [CI/CD](#5-cicd)
6. [릴리스 자동화](#6-릴리스-자동화)
7. [요약](#7-요약)

---

## 1. 빌드 기초

### 1.1 빌드 명령

```bash
# Basic build
go build -o myapp ./cmd/server

# Build with optimizations
go build -ldflags="-s -w" -o myapp ./cmd/server
# -s: strip symbol table
# -w: strip DWARF debug info
# Reduces binary size ~30%

# Build all packages
go build ./...

# Install binary to $GOPATH/bin
go install ./cmd/server

# Build for specific package
go build -o api ./cmd/api
go build -o worker ./cmd/worker
```

### 1.2 빌드 시점 변수

```go
// main.go
var (
    version = "dev"
    commit  = "unknown"
    date    = "unknown"
    builtBy = "unknown"
)

func main() {
    if len(os.Args) > 1 && os.Args[1] == "version" {
        fmt.Printf("Version: %s\nCommit:  %s\nDate:    %s\nBuilt:   %s\n",
            version, commit, date, builtBy)
        return
    }
    // ...
}
```

```bash
go build -ldflags "\
  -X main.version=1.2.3 \
  -X main.commit=$(git rev-parse --short HEAD) \
  -X main.date=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -X main.builtBy=$(whoami)" \
  -o myapp ./cmd/server
```

### 1.3 빌드 태그

```go
//go:build production

package config

const Debug = false
const LogLevel = "warn"
```

```bash
go build -tags production -o myapp
```

---

## 2. 크로스 컴파일

### 2.1 플랫폼 타겟

```bash
# Linux AMD64 (most common server target)
GOOS=linux GOARCH=amd64 go build -o myapp-linux-amd64

# Linux ARM64 (AWS Graviton, Raspberry Pi 4)
GOOS=linux GOARCH=arm64 go build -o myapp-linux-arm64

# macOS Apple Silicon
GOOS=darwin GOARCH=arm64 go build -o myapp-darwin-arm64

# macOS Intel
GOOS=darwin GOARCH=amd64 go build -o myapp-darwin-amd64

# Windows
GOOS=windows GOARCH=amd64 go build -o myapp-windows-amd64.exe

# List all supported platforms
go tool dist list
```

### 2.2 정적 바이너리

```bash
# Fully static binary (no libc dependency)
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o myapp ./cmd/server

# Verify it's static
file myapp
# myapp: ELF 64-bit LSB executable, x86-64, statically linked

ldd myapp
# not a dynamic executable
```

### 2.3 빌드 스크립트

```bash
#!/bin/bash
# build.sh — Cross-compile for all targets

VERSION=$(git describe --tags --always --dirty)
COMMIT=$(git rev-parse --short HEAD)
DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
LDFLAGS="-s -w -X main.version=${VERSION} -X main.commit=${COMMIT} -X main.date=${DATE}"

PLATFORMS=(
    "linux/amd64"
    "linux/arm64"
    "darwin/amd64"
    "darwin/arm64"
    "windows/amd64"
)

for PLATFORM in "${PLATFORMS[@]}"; do
    GOOS=${PLATFORM%/*}
    GOARCH=${PLATFORM#*/}
    OUTPUT="dist/myapp-${GOOS}-${GOARCH}"
    [[ "$GOOS" == "windows" ]] && OUTPUT+=".exe"

    echo "Building ${OUTPUT}..."
    CGO_ENABLED=0 GOOS=$GOOS GOARCH=$GOARCH \
        go build -ldflags="${LDFLAGS}" -o "${OUTPUT}" ./cmd/server
done
```

---

## 3. Docker

### 3.1 멀티 스테이지 빌드

```dockerfile
# Stage 1: Build
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Cache dependencies
COPY go.mod go.sum ./
RUN go mod download

# Build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o /app/server ./cmd/server

# Stage 2: Runtime
FROM scratch

# Copy binary
COPY --from=builder /app/server /server

# Copy CA certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Run as non-root
USER 1000:1000

EXPOSE 8080

ENTRYPOINT ["/server"]
```

### 3.2 Docker Compose

```yaml
version: "3.8"

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgres://postgres:secret@db:5432/myapp?sslmode=disable
      - PORT=8080
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: myapp
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

### 3.3 Distroless 대안

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /server ./cmd/server

FROM gcr.io/distroless/static-debian12
COPY --from=builder /server /server
USER nonroot:nonroot
ENTRYPOINT ["/server"]
```

---

## 4. 파일 임베딩

### 4.1 embed 패키지 (Go 1.16+)

```go
package main

import (
    "embed"
    "fmt"
    "io/fs"
    "net/http"
)

//go:embed static/*
var staticFiles embed.FS

//go:embed templates/*.html
var templateFiles embed.FS

//go:embed VERSION
var version string

//go:embed config/defaults.json
var defaultConfig []byte

func main() {
    fmt.Println("Version:", version)
    fmt.Println("Config:", string(defaultConfig))

    // Serve embedded static files
    staticFS, _ := fs.Sub(staticFiles, "static")
    http.Handle("/static/", http.StripPrefix("/static/",
        http.FileServer(http.FS(staticFS))))

    http.ListenAndServe(":8080", nil)
}
```

---

## 5. CI/CD

### 5.1 GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'

      - name: Download dependencies
        run: go mod download

      - name: Vet
        run: go vet ./...

      - name: Test
        run: go test -race -coverprofile=coverage.out ./...

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.out

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'

      - name: Build
        run: |
          CGO_ENABLED=0 go build -ldflags="-s -w" -o myapp ./cmd/server

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: myapp
          path: myapp
```

---

## 6. 릴리스 자동화

### 6.1 GoReleaser

```yaml
# .goreleaser.yml
project_name: myapp

builds:
  - main: ./cmd/server
    binary: myapp
    env:
      - CGO_ENABLED=0
    goos:
      - linux
      - darwin
      - windows
    goarch:
      - amd64
      - arm64
    ldflags:
      - -s -w
      - -X main.version={{.Version}}
      - -X main.commit={{.Commit}}
      - -X main.date={{.Date}}

archives:
  - format: tar.gz
    name_template: "{{ .ProjectName }}_{{ .Version }}_{{ .Os }}_{{ .Arch }}"
    format_overrides:
      - goos: windows
        format: zip

dockers:
  - image_templates:
      - "ghcr.io/user/myapp:{{ .Version }}"
      - "ghcr.io/user/myapp:latest"
    dockerfile: Dockerfile

changelog:
  sort: asc
  filters:
    exclude:
      - "^docs:"
      - "^test:"
```

```bash
# Local test
goreleaser release --snapshot --clean

# Tag and release
git tag v1.0.0
git push origin v1.0.0
# CI triggers goreleaser
```

---

## 7. 요약

### 핵심 포인트

1. **단일 정적 바이너리** — `CGO_ENABLED=0 go build`로 의존성 없는 실행 파일을 생성한다.
2. **빌드 정보를 위한 `ldflags`** — 빌드 시점에 버전, 커밋, 날짜를 주입한다.
3. **멀티 스테이지 Docker** — golang 이미지에서 빌드하고, scratch/distroless에서 실행한다.
4. **에셋을 위한 `embed`** — 정적 파일, 템플릿, 설정을 바이너리에 포함한다.
5. **쉬운 크로스 컴파일** — 크로스 컴파일러 없이 `GOOS`/`GOARCH`만으로 모든 플랫폼을 지원한다.
6. **자동화를 위한 GoReleaser** — 크로스 컴파일, 아카이브, Docker 이미지, 변경 로그를 처리한다.

---

## 연습 문제

### 연습 1: 빌드 파이프라인
`build`, `test`, `lint`, `docker`, `clean` 타겟이 있는 Makefile을 만든다. 버전 주입을 포함한다.

### 연습 2: 멀티 플랫폼 Docker
`docker buildx`를 사용하여 다중 아키텍처 이미지(amd64 + arm64)를 생성하는 Docker 빌드를 만든다.

### 연습 3: 임베디드 웹 앱
`//go:embed`를 사용하여 모든 HTML, CSS, JS, 이미지 에셋이 바이너리에 포함된 웹 애플리케이션을 만든다.

### 연습 4: 릴리스 워크플로우
태그가 푸시되면 빌드, 테스트를 수행하고 3개 플랫폼용 바이너리로 GitHub 릴리스를 생성하는 GitHub Actions 워크플로우를 구축한다.
