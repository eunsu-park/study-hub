# Multi-Stage Build Patterns

**Previous**: [Persistent Volumes](./13_Persistent_Volumes.md) | **Next**: [Podman and OCI](./15_Podman_and_OCI.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain multi-stage build fundamentals and why they produce smaller, more secure images
2. Apply the builder pattern for compiled languages such as Go, Rust, and C++
3. Optimize Docker image size using selective COPY from build stages
4. Leverage BuildKit cache mounts and other performance enhancements
5. Use distroless and scratch base images for minimal production images
6. Manage build arguments, target stages, and conditional builds
7. Implement real-world Dockerfile patterns for Node.js, Python, and Java applications

## Table of Contents

Before the patterns reference, read [**Theory & Principles**](#theory--principles) — the layer-cache key algorithm that drives every cache hit decision, BuildKit's DAG execution that parallelizes stages and skips unused outputs, and the multi-platform fan-out that turns one build into a manifest list.

1. [Multi-Stage Build Fundamentals](#1-multi-stage-build-fundamentals)
2. [Builder Pattern for Compiled Languages](#2-builder-pattern-for-compiled-languages)
3. [Optimizing Image Size](#3-optimizing-image-size)
4. [BuildKit Cache Strategies](#4-buildkit-cache-strategies)
5. [BuildKit Features and Enhancements](#5-buildkit-features-and-enhancements)
6. [Distroless and Scratch Images](#6-distroless-and-scratch-images)
7. [Build Arguments and Target Stages](#7-build-arguments-and-target-stages)
8. [Real-World Dockerfile Patterns](#8-real-world-dockerfile-patterns)
9. [Advanced Patterns](#9-advanced-patterns)
10. [Practice Exercises](#10-practice-exercises)

**Difficulty**: ⭐⭐⭐

---

Before multi-stage builds, developers faced a dilemma: either ship large images containing build tools, compilers, and source code, or maintain complex scripts that built artifacts outside Docker and copied them in. Multi-stage builds elegantly solve this by allowing multiple `FROM` instructions in a single Dockerfile, each starting a new build stage. Only the final stage becomes the shipped image, while intermediate stages provide build tools and artifacts that are discarded.

---

## Theory & Principles

Multi-stage builds and BuildKit are two answers to the same question: "given a Dockerfile, what is the minimum amount of work that produces the right artifact?" The "right artifact" part means a small, secure, multi-architecture-capable image. The "minimum work" part is the layer-cache algorithm running over a graph of stages and instructions, parallelizing what it can. Once you understand cache keys (when does a layer get reused?), the DAG model (what runs in parallel?), and the multi-platform fan-out (how does one build produce arm64 + amd64?), every "why is my build slow / fat / wrong arch?" question has a tractable answer.

### A. The Layer Cache Key Algorithm — In Detail

A Dockerfile is a sequence of instructions, each producing a layer. For each instruction, BuildKit (and the legacy builder) computes a **cache key** and looks for an existing layer with that key in the local store and any configured remote caches.

The cache key is a hash of:

| Instruction | Inputs to the hash |
|-------------|--------------------|
| `FROM image` | The full digest of the resolved image (`@sha256:...`). A tag like `node:20` is resolved to a digest first. |
| `RUN command` | The command text (verbatim) + the digest of the parent layer + relevant build args. |
| `COPY src dst` | The content hash of every file in `src` after `.dockerignore` filtering + the destination path + the parent digest. |
| `ARG name` | The arg name (the value matters only when `RUN` references the arg). |
| Metadata-only (`ENV`, `LABEL`, `EXPOSE`, `WORKDIR`) | The instruction text + parent digest. |

Two key consequences:

1. **The chain breaks at the first miss.** If instruction N changes (or its inputs change), instructions N+1, N+2, ... all miss too because each carries the previous layer's digest in its key. This is why "put stable instructions first" is the iron rule of Dockerfile optimization.
2. **`COPY` is content-hashed, not name-hashed.** Editing `.gitignore` does not invalidate `COPY package.json /app/`; editing `package.json` does. Splitting "copy dependency manifests" from "copy source code" is the standard way to keep the dependency-install layer cached when only source changes.

Splitting `COPY` is so important it deserves an explicit example:

```dockerfile
# Bad: any source change busts npm install
COPY . /app
RUN npm ci

# Good: only package.json/package-lock.json changes bust npm install
COPY package.json package-lock.json /app/
RUN npm ci
COPY . /app
```

The first form rebuilds `node_modules` (slow) on every commit. The second form only rebuilds it when dependencies actually change.

### B. BuildKit's DAG Execution

The legacy builder runs instructions sequentially: instruction N must complete before instruction N+1 begins, even if they don't depend on each other. **BuildKit** (the default since Docker 23.0) parses the Dockerfile into a **directed acyclic graph (DAG)** where each instruction is a node and edges express data dependencies. Independent nodes run concurrently.

For a single-stage Dockerfile this barely matters. For a multi-stage Dockerfile it matters a lot:

```dockerfile
FROM node:20 AS frontend
WORKDIR /src
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM golang:1.22 AS backend
WORKDIR /src
COPY backend/go.mod backend/go.sum ./
RUN go mod download
COPY backend/ ./
RUN go build -o server .

FROM gcr.io/distroless/static AS final
COPY --from=frontend /src/dist /app/static
COPY --from=backend /src/server /app/server
ENTRYPOINT ["/app/server"]
```

The legacy builder runs `frontend` to completion, then `backend`, then assembles `final` — sequential. BuildKit sees that `frontend` and `backend` have no edges between them and runs both concurrently. On a multi-core machine the wall time roughly halves.

BuildKit also **skips unused stages**. If you target `final`, BuildKit walks back from `final`, finds it depends on `frontend` and `backend`, and runs only those. A "test" stage that final does not depend on is not built unless you `docker build --target test .`.

### C. BuildKit Mounts: Cache, Bind, and Secret

A `RUN` instruction in BuildKit can opt into **mounts** that change what the command sees without changing the layer it produces:

- **`--mount=type=cache,target=/root/.cache/pip`** — a writable cache directory that *persists between builds* but does *not* end up in the resulting layer. The next `pip install` finds previously downloaded wheels; the resulting image still does not contain them. Same idea for `~/.npm`, `~/.cargo/registry`, `/var/cache/apt`. The single biggest speed-up for repeated builds.
- **`--mount=type=bind,source=.,target=/src`** — read-only bind of the build context (or a previous stage) into the command, without baking it into a layer. Useful when a build step needs to *read* large amounts of source but should not produce a layer with all of it.
- **`--mount=type=secret,id=mytoken`** — file appears at `/run/secrets/mytoken` for the duration of the `RUN`, sourced from `--secret id=mytoken,src=./token.txt` at build time. Never written to a layer; safe for npm tokens, private-registry credentials, etc.
- **`--mount=type=ssh`** — forwards the host's SSH agent into the `RUN`. Lets `git clone git@github.com:...` work for private repos without baking SSH keys.

These mounts are the single biggest reason "modern" Dockerfiles look different from "classic" Dockerfiles. Used well, build times collapse and secrets stop ending up in layers.

### D. Multi-Platform Builds: One Build, Many Architectures

A single `docker build` produces one image for one architecture (the host's). For an image that must run on both AMD64 servers and ARM64 (Apple Silicon, AWS Graviton) you need two builds, then a manifest list pointing at both.

`docker buildx` automates this:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t myregistry/myapp:1.0 --push .
```

What happens:

1. BuildKit fans out: one build per requested platform.
2. Each platform build runs in an isolated environment that emulates (via QEMU) or uses a native builder for the target architecture. Building arm64 from an amd64 host typically uses QEMU userspace emulation (slow but works) or remote arm64 builders (fast).
3. Each build produces a per-platform image with its own manifest and layers.
4. After all builds succeed, buildx pushes a **manifest list** (OCI image index) to the registry. The list points at each per-platform manifest by digest.
5. When users `docker pull myregistry/myapp:1.0`, the client picks the manifest matching the host platform from the list automatically.

`docker buildx ls` shows your builders. The default `docker-container` driver runs BuildKit in a container; you can also configure remote builders for native arm64 builds.

### E. Stage Targeting and Conditional Builds

`docker build --target <stage>` builds only up to the named stage and stops. Used for:

- **Test stages.** Define a `test` stage that runs `pytest`/`go test`/`npm test`. CI runs `docker build --target test .` to fail the pipeline on test failure. Production images target `prod` and never build the test stage.
- **Lint stages.** Same idea for `eslint`, `golangci-lint`, etc.
- **Debug images.** A `debug` stage adds tools (`bash`, `curl`, `tcpdump`) on top of the production stage. Built only when explicitly targeted.

Combined with `ARG` and conditional logic, a single Dockerfile can produce dev, test, prod, and debug images by varying `--target` and `--build-arg`. The cache is shared across them — common stages built once, divergent stages built only as needed.

### F. Image Bases Revisited: How "Small" Different Bases Are

The runtime stage's base image determines the floor of image size and the surface area of attack:

| Base | Approx size | Tools available | Best for |
|------|-------------|-----------------|----------|
| `ubuntu:22.04` | ~80 MB | Full Debian-derived userspace | Dev, debugging, rare cases that genuinely need bash + apt |
| `debian:bookworm-slim` | ~30 MB | Minimal Debian | General purpose |
| `alpine:3.19` | ~7 MB | musl libc, busybox, apk | When the language toolchain works with musl (Go does, Python sometimes doesn't) |
| `gcr.io/distroless/python3` | ~50 MB | Just Python + glibc | Production Python — no shell, no apt, no curl |
| `gcr.io/distroless/static` | ~2 MB | Just glibc-free static binaries | Production Go / Rust / Zig binaries |
| `scratch` | 0 MB | Nothing | Static binaries that bring everything they need |

Distroless and scratch are the compelling endpoints — production images measured in single-digit MB, with no shell to be exploited, no package manager to install backdoors with. The trade-off is debugging: `kubectl exec -it ... sh` does not work because there's no shell. Modern Kubernetes adds **ephemeral debug containers** (`kubectl debug`) that attach a debug image to the same Pod's namespaces, which closes the gap.

### From Theory to the Patterns Below

- **Builder pattern** (§A, §B, §C) — first stage compiles, last stage runs. `COPY --from=builder /app/binary /` is the seam.
- **`COPY` ordering** — manifests then code (§A) keeps dependency layers cached.
- **`RUN --mount=type=cache,...`** (§C) — keep dependency-fetch caches between builds without bloating layers.
- **`RUN --mount=type=secret,...`** (§C) — credentials at build time without leaving them in a layer.
- **Multi-target Dockerfile** (§E) — one file, multiple `--target` values for dev/test/prod/debug.
- **`docker buildx --platform linux/amd64,linux/arm64`** (§D) — one command, manifest list output.
- **Distroless / scratch final stages** (§F) — minimum runtime size and attack surface.

The remaining sections walk these patterns. Whenever a build runs slowly, profile with `--progress=plain` and check which instructions cache-miss; the answer is almost always upstream of where you noticed the pain.

---

## 1. Multi-Stage Build Fundamentals

### The Problem: Fat Images

A traditional single-stage Dockerfile for a Go application:

```dockerfile
# Single-stage: 800MB+ image
FROM golang:1.22
WORKDIR /app
COPY . .
RUN go build -o server .
CMD ["./server"]
```

This image includes the entire Go toolchain, source code, and all build dependencies -- none of which are needed at runtime.

### The Solution: Multi-Stage Builds

```dockerfile
# Stage 1: Build
FROM golang:1.22 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o server .

# Stage 2: Runtime (final image)
FROM alpine:3.19
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /app/server .
CMD ["./server"]
```

```
┌─────────────────────────────────────────────────────────────┐
│                  Multi-Stage Build Flow                       │
│                                                              │
│  Stage 1 (builder)              Stage 2 (final)             │
│  ┌─────────────────────┐       ┌─────────────────────┐     │
│  │ golang:1.22          │       │ alpine:3.19          │     │
│  │ ┌─────────────────┐ │       │ ┌─────────────────┐ │     │
│  │ │ Go toolchain    │ │       │ │ ca-certificates  │ │     │
│  │ │ Source code      │ │       │ │                  │ │     │
│  │ │ Dependencies     │ │  COPY │ │ server binary    │ │     │
│  │ │ ─────────────── │ │──────►│ │ (from builder)   │ │     │
│  │ │ server binary ★ │ │       │ └─────────────────┘ │     │
│  │ └─────────────────┘ │       │                      │     │
│  │ Size: ~800MB         │       │ Size: ~15MB          │     │
│  └─────────────────────┘       └─────────────────────┘     │
│       ↓ discarded                    ↓ shipped              │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts

- Each `FROM` instruction starts a new stage
- Stages can be named with `AS <name>`
- `COPY --from=<stage>` copies files from a previous stage
- Only the final stage is included in the built image
- Intermediate stages are cached but not shipped

---

## 2. Builder Pattern for Compiled Languages

### Go

```dockerfile
FROM golang:1.22-alpine AS builder
RUN apk add --no-cache git
WORKDIR /app

# Cache dependencies
COPY go.mod go.sum ./
RUN go mod download

# Build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o /app/server ./cmd/server

# Final image
FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /app/server /server
ENTRYPOINT ["/server"]
```

Result: ~5-10MB image containing only the static binary and CA certificates.

### Rust

```dockerfile
FROM rust:1.77 AS builder
WORKDIR /app

# Cache dependencies with a dummy build
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Build the actual application
COPY src ./src
RUN touch src/main.rs  # Ensure rebuild
RUN cargo build --release

# Final image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/myapp /usr/local/bin/
CMD ["myapp"]
```

### C++

```dockerfile
FROM gcc:13 AS builder
WORKDIR /app
COPY . .
RUN mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# Final image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/build/myapp /usr/local/bin/
CMD ["myapp"]
```

---

## 3. Optimizing Image Size

### Layer Optimization

```dockerfile
# BAD: Each RUN creates a separate layer
RUN apt-get update
RUN apt-get install -y curl wget git
RUN apt-get clean

# GOOD: Single layer with cleanup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl wget git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*
```

### Selective COPY

```dockerfile
FROM node:20 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Only copy what's needed for production
FROM node:20-slim
WORKDIR /app
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
CMD ["node", "dist/index.js"]
```

### Image Size Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                Image Size Comparison (Go App)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  golang:1.22         ███████████████████████████████  ~820MB │
│  golang:1.22-alpine  █████████████                   ~260MB │
│  alpine + binary     █                               ~15MB  │
│  distroless          ▌                               ~8MB   │
│  scratch + binary    ▏                               ~5MB   │
│                                                              │
│  ──────────────────────────────────────────────────────────  │
│  Key insight: 99% reduction from full image to scratch       │
└─────────────────────────────────────────────────────────────┘
```

### .dockerignore

Always use `.dockerignore` to exclude unnecessary files from the build context:

```
# .dockerignore
.git
.gitignore
node_modules
*.md
LICENSE
.env
.env.*
docker-compose*.yml
Dockerfile
.dockerignore
__pycache__
*.pyc
.pytest_cache
.coverage
dist
build
```

---

## 4. BuildKit Cache Strategies

### Dependency Cache Mounts

Cache mounts preserve package manager caches across builds, dramatically speeding up rebuilds:

```dockerfile
# syntax=docker/dockerfile:1

# Python with pip cache
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim
COPY --from=builder /install /usr/local
COPY . /app
CMD ["python", "/app/main.py"]
```

```dockerfile
# Node.js with npm cache
FROM node:20 AS builder
WORKDIR /app
COPY package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci
COPY . .
RUN npm run build
```

```dockerfile
# Go with module cache
FROM golang:1.22 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download
COPY . .
RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -o /app/server .
```

### Secret Mounts

Securely pass secrets during build without embedding them in layers:

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim
RUN --mount=type=secret,id=pip_conf,target=/etc/pip.conf \
    pip install private-package
```

```bash
# Build with secret
docker build --secret id=pip_conf,src=./pip.conf .
```

---

## 5. BuildKit Features and Enhancements

### Enabling BuildKit

```bash
# Environment variable
export DOCKER_BUILDKIT=1
docker build .

# Or use docker buildx
docker buildx build .

# Docker Compose
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose build
```

### Parallel Stage Building

BuildKit automatically builds independent stages in parallel:

```dockerfile
# These stages build in parallel
FROM golang:1.22 AS backend-builder
WORKDIR /app
COPY backend/ .
RUN go build -o /backend ./cmd/server

FROM node:20 AS frontend-builder
WORKDIR /app
COPY frontend/ .
RUN npm ci && npm run build

# Final stage uses artifacts from both
FROM nginx:alpine
COPY --from=backend-builder /backend /usr/local/bin/
COPY --from=frontend-builder /app/dist /usr/share/nginx/html
```

```
┌─────────────────────────────────────────────────────────────┐
│              BuildKit Parallel Build                          │
│                                                              │
│  Time ──────────────────────────────────────────────►        │
│                                                              │
│  Without BuildKit (sequential):                              │
│  [backend-builder ████████████][frontend-builder ████████]   │
│  Total: ~8 min                                               │
│                                                              │
│  With BuildKit (parallel):                                   │
│  [backend-builder  ████████████]                             │
│  [frontend-builder ████████    ]                             │
│  Total: ~5 min                                               │
└─────────────────────────────────────────────────────────────┘
```

### Build Output Modes

```bash
# Plain text output (for CI logs)
docker buildx build --progress=plain .

# TTY output (default, interactive)
docker buildx build --progress=auto .

# Export build to OCI tarball
docker buildx build --output type=oci,dest=image.tar .

# Export filesystem without creating image
docker buildx build --output type=local,dest=./output .
```

---

## 6. Distroless and Scratch Images

### scratch

`scratch` is a special empty image -- the absolute minimal base:

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o /server .

# scratch = empty image, nothing at all
FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
```

Limitations of scratch:
- No shell (cannot `docker exec sh`)
- No package manager
- No timezone data, CA certificates (must copy manually)
- No user management (`/etc/passwd` missing)

### Distroless (Google)

Distroless images contain only the application runtime -- no shell, package manager, or OS utilities:

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o /server .

# Distroless: minimal runtime with CA certs, timezone data
FROM gcr.io/distroless/static-debian12
COPY --from=builder /server /server
USER nonroot:nonroot
ENTRYPOINT ["/server"]
```

Available distroless images:
- `gcr.io/distroless/static-debian12` -- statically compiled binaries (Go)
- `gcr.io/distroless/base-debian12` -- dynamically linked binaries (Rust, C++)
- `gcr.io/distroless/cc-debian12` -- requires libstdc++
- `gcr.io/distroless/java21-debian12` -- Java applications
- `gcr.io/distroless/python3-debian12` -- Python applications
- `gcr.io/distroless/nodejs22-debian12` -- Node.js applications

### Creating a Non-Root User in scratch

```dockerfile
FROM golang:1.22 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o /server .

# Create minimal passwd file
RUN echo "appuser:x:10001:10001::/nonexistent:/sbin/nologin" > /etc/passwd.app

FROM scratch
COPY --from=builder /etc/passwd.app /etc/passwd
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /server /server
USER appuser
ENTRYPOINT ["/server"]
```

---

## 7. Build Arguments and Target Stages

### Build Arguments

```dockerfile
# Parameterize the base image
ARG GO_VERSION=1.22
FROM golang:${GO_VERSION} AS builder
ARG APP_VERSION=dev
WORKDIR /app
COPY . .
RUN go build -ldflags="-X main.version=${APP_VERSION}" -o /server .

FROM alpine:3.19
COPY --from=builder /server /server
CMD ["/server"]
```

```bash
# Override defaults
docker build \
  --build-arg GO_VERSION=1.21 \
  --build-arg APP_VERSION=1.2.3 \
  -t myapp:1.2.3 .
```

### Target Stages

Build only up to a specific stage:

```dockerfile
FROM node:20 AS base
WORKDIR /app
COPY package*.json ./

FROM base AS dependencies
RUN npm ci

FROM dependencies AS test
COPY . .
RUN npm test

FROM dependencies AS build
COPY . .
RUN npm run build

FROM node:20-slim AS production
WORKDIR /app
COPY --from=build /app/dist ./dist
COPY --from=dependencies /app/node_modules ./node_modules
CMD ["node", "dist/index.js"]
```

```bash
# Build only the test stage
docker build --target test -t myapp:test .

# Build only the production stage (default: last stage)
docker build --target production -t myapp:prod .

# Build for development
docker build --target dependencies -t myapp:dev .
```

### Conditional Stages

```dockerfile
ARG BUILD_ENV=production

FROM node:20 AS base
WORKDIR /app
COPY package*.json ./
RUN npm ci

FROM base AS development
COPY . .
CMD ["npm", "run", "dev"]

FROM base AS production-build
COPY . .
RUN npm run build

FROM node:20-slim AS production
WORKDIR /app
COPY --from=production-build /app/dist ./dist
COPY --from=production-build /app/node_modules ./node_modules
CMD ["node", "dist/index.js"]
```

---

## 8. Real-World Dockerfile Patterns

### Node.js (Full-Stack Application)

```dockerfile
# syntax=docker/dockerfile:1
FROM node:20-alpine AS base
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Install dependencies
FROM base AS deps
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production && \
    cp -R node_modules /prod_modules && \
    npm ci

# Build
FROM base AS build
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Production
FROM base AS production
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 appuser
COPY --from=build /app/dist ./dist
COPY --from=deps /prod_modules ./node_modules
USER appuser
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

### Python (FastAPI Application)

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Build stage: compile dependencies
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install --no-warn-script-location \
    -r requirements.txt

# Production
FROM base AS production
COPY --from=builder /install /usr/local
RUN useradd --create-home --shell /bin/bash appuser
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Java (Spring Boot Application)

```dockerfile
# syntax=docker/dockerfile:1
FROM eclipse-temurin:21-jdk AS builder
WORKDIR /app
COPY gradle/ gradle/
COPY gradlew build.gradle.kts settings.gradle.kts ./
RUN --mount=type=cache,target=/root/.gradle \
    ./gradlew dependencies --no-daemon
COPY src ./src
RUN --mount=type=cache,target=/root/.gradle \
    ./gradlew bootJar --no-daemon

# Extract Spring Boot layers for better caching
FROM eclipse-temurin:21-jdk AS extractor
WORKDIR /app
COPY --from=builder /app/build/libs/*.jar app.jar
RUN java -Djarmode=layertools -jar app.jar extract

# Production
FROM eclipse-temurin:21-jre-alpine
RUN addgroup --system spring && adduser --system --ingroup spring spring
WORKDIR /app
COPY --from=extractor /app/dependencies/ ./
COPY --from=extractor /app/spring-boot-loader/ ./
COPY --from=extractor /app/snapshot-dependencies/ ./
COPY --from=extractor /app/application/ ./
USER spring
EXPOSE 8080
ENTRYPOINT ["java", "org.springframework.boot.loader.launch.JarLauncher"]
```

---

## 9. Advanced Patterns

### Cross-Compilation

```dockerfile
# Build for multiple architectures
FROM --platform=$BUILDPLATFORM golang:1.22 AS builder
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -o /server .

FROM alpine:3.19
COPY --from=builder /server /server
CMD ["/server"]
```

```bash
# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myapp:latest \
  --push .
```

### Copying from External Images

```dockerfile
FROM nginx:alpine

# Copy a binary from a completely separate image
COPY --from=busybox:latest /bin/wget /usr/local/bin/wget

# Copy configuration from a custom image
COPY --from=mycompany/nginx-config:latest /etc/nginx/nginx.conf /etc/nginx/
```

### Testing in Multi-Stage Builds

```dockerfile
FROM python:3.12-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Test stage
FROM base AS test
RUN pip install pytest pytest-cov
RUN pytest --cov=app tests/

# Lint stage
FROM base AS lint
RUN pip install ruff
RUN ruff check .

# Production (only built if test and lint pass with --target)
FROM base AS production
USER nobody
CMD ["python", "main.py"]
```

---

## 10. Practice Exercises

### Exercise 1: Basic Multi-Stage Build (Beginner)

Convert this single-stage Dockerfile into a multi-stage build:

```dockerfile
FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python -m pytest tests/
CMD ["python", "app.py"]
```

<details>
<summary>Solution</summary>

```dockerfile
# Test stage
FROM python:3.12-slim AS test
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python -m pytest tests/

# Production stage
FROM python:3.12-slim AS production
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY src/ ./src/
USER nobody
CMD ["python", "app.py"]
```

</details>

### Exercise 2: Go Application with Scratch (Intermediate)

Write a multi-stage Dockerfile for a Go web server that:
- Builds a statically linked binary
- Uses scratch as the final image
- Includes CA certificates for HTTPS
- Runs as a non-root user

<details>
<summary>Solution</summary>

```dockerfile
FROM golang:1.22-alpine AS builder
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /server .
RUN echo "appuser:x:10001:10001::/nonexistent:/sbin/nologin" > /etc/passwd.min

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /etc/passwd.min /etc/passwd
COPY --from=builder /server /server
USER appuser
EXPOSE 8080
ENTRYPOINT ["/server"]
```

</details>

### Exercise 3: Full-Stack Build (Advanced)

Create a multi-stage Dockerfile for an application with:
- A React frontend (npm build)
- A Go backend API
- Nginx serving the frontend and proxying API requests
- All built in parallel stages

<details>
<summary>Solution</summary>

```dockerfile
# syntax=docker/dockerfile:1

# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Build backend (runs in parallel with Stage 1)
FROM golang:1.22-alpine AS backend
WORKDIR /app
COPY backend/go.mod backend/go.sum ./
RUN go mod download
COPY backend/ .
RUN CGO_ENABLED=0 go build -o /api-server .

# Stage 3: Final image
FROM nginx:alpine
# Copy frontend build
COPY --from=frontend /app/dist /usr/share/nginx/html
# Copy backend binary
COPY --from=backend /api-server /usr/local/bin/
# Copy nginx config with API proxy
COPY nginx.conf /etc/nginx/nginx.conf
# Startup script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]
```

</details>

### Exercise 4: Cache Optimization (Advanced)

Take this Dockerfile and optimize it using BuildKit cache mounts, proper layer ordering, and multi-stage builds to achieve the fastest possible rebuild time:

```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python -m pytest
RUN python setup.py bdist_wheel
CMD ["python", "-m", "myapp"]
```

<details>
<summary>Solution</summary>

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base
WORKDIR /app

# Dependencies (cached unless requirements change)
FROM base AS deps
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

# Test stage
FROM base AS test
COPY --from=deps /install /usr/local
COPY . .
RUN python -m pytest

# Build wheel
FROM base AS build
COPY --from=deps /install /usr/local
COPY . .
RUN python setup.py bdist_wheel

# Production
FROM python:3.12-slim AS production
COPY --from=build /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
USER nobody
CMD ["python", "-m", "myapp"]
```

</details>

---

**Previous**: [Persistent Volumes](./13_Persistent_Volumes.md) | **Next**: [Podman and OCI](./15_Podman_and_OCI.md)
