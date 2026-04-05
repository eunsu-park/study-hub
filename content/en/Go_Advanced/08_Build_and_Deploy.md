# 19. Build and Deploy

**Previous**: [Performance Profiling](./07_Performance_Profiling.md) | **Next**: [Network Programming](./09_Network_Programming.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Cross-compile Go binaries for multiple platforms
2. Create minimal Docker images with multi-stage builds
3. Embed files and set build-time variables with `ldflags`
4. Set up CI/CD pipelines for Go projects
5. Use GoReleaser for automated release management

---

One of Go's greatest strengths is deployment simplicity. A Go program compiles to a single static binary with no runtime dependencies. This makes containerization, cross-compilation, and distribution straightforward.

## Table of Contents
1. [Build Fundamentals](#1-build-fundamentals)
2. [Cross-Compilation](#2-cross-compilation)
3. [Docker](#3-docker)
4. [File Embedding](#4-file-embedding)
5. [CI/CD](#5-cicd)
6. [Release Automation](#6-release-automation)
7. [Summary](#7-summary)

---

## 1. Build Fundamentals

### 1.1 Build Commands

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

### 1.2 Build-Time Variables

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

### 1.3 Build Tags

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

## 2. Cross-Compilation

### 2.1 Platform Targets

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

### 2.2 Static Binary

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

### 2.3 Build Script

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

### 3.1 Multi-Stage Build

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

### 3.3 Distroless Alternative

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

## 4. File Embedding

### 4.1 embed Package (Go 1.16+)

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

## 6. Release Automation

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

## 7. Summary

### Key Takeaways

1. **Single static binary** — `CGO_ENABLED=0 go build` produces zero-dependency executables.
2. **`ldflags` for build info** — inject version, commit, date at build time.
3. **Multi-stage Docker** — build in golang image, run from scratch/distroless.
4. **`embed` for assets** — include static files, templates, and configs in the binary.
5. **Cross-compile easily** — `GOOS`/`GOARCH` for any platform without a cross-compiler.
6. **GoReleaser for automation** — handles cross-compilation, archives, Docker images, and changelogs.

---

## Exercises

### Exercise 1: Build Pipeline
Create a Makefile with targets: `build`, `test`, `lint`, `docker`, `clean`. Include version injection.

### Exercise 2: Multi-Platform Docker
Create a Docker build that produces multi-architecture images (amd64 + arm64) using `docker buildx`.

### Exercise 3: Embedded Web App
Build a web application where all HTML, CSS, JS, and image assets are embedded in the binary using `//go:embed`.

### Exercise 4: Release Workflow
Set up a GitHub Actions workflow that builds, tests, and creates GitHub releases with binaries for 3 platforms when a tag is pushed.
