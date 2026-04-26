# Dockerfile

**Previous**: [Docker Images and Containers](./02_Images_and_Containers.md) | **Next**: [Docker Compose](./04_Docker_Compose.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a Dockerfile is and why it provides reproducible, version-controlled image builds
2. Write Dockerfiles using core instructions: FROM, WORKDIR, COPY, RUN, CMD, EXPOSE, and ENV
3. Distinguish between CMD and ENTRYPOINT, and between COPY and ADD
4. Apply multi-stage builds to separate build and runtime environments and reduce image size
5. Implement best practices including .dockerignore, layer caching, small base images, and non-root users
6. Build Docker images using docker build with tags, build arguments, and cache control

---

Before the syntax tour, read [**Theory & Principles**](#theory--principles) — how the build context is shipped to the daemon, how each instruction becomes a layer, and how BuildKit's graph-based execution decides cache hits and parallelism.

While pulling pre-built images from Docker Hub is convenient, real-world projects require custom images tailored to your specific application and dependencies. The Dockerfile is the standard mechanism for defining these custom images as code. By learning Dockerfile syntax and best practices such as multi-stage builds and layer optimization, you gain full control over your application's packaging and can ensure consistent, secure, and efficient container images.

---

## Theory & Principles

A Dockerfile is just a text file. The interesting machinery is what happens when you run `docker build` against it: the **build context** is shipped to the engine, each **instruction** is interpreted as a transformation that produces a new image layer, the **cache** decides which transformations can be skipped, and **BuildKit** turns the linear instruction list into a parallel directed graph. Understanding these four pieces is what separates Dockerfiles that build in five seconds from those that build in five minutes.

### A. The Build Context: What Gets Sent to the Daemon

`docker build .` does not magically operate on your working directory. It does this:

1. The CLI walks the directory passed as the build context (the trailing `.`).
2. It reads `.dockerignore` and excludes matching paths.
3. It tarballs everything that remains and streams it over the daemon socket.
4. The daemon (which may be on a different machine entirely) starts building from that tarball.

Two consequences follow immediately:

- **Anything outside the context is invisible.** A `COPY ../config /etc` is a syntax error — `..` is not in the tarball. The build cannot reach files the CLI did not pack.
- **Bigger context = slower build.** A 2 GB `node_modules` directory takes seconds just to upload, even before any instruction runs. `.dockerignore` is not a "nice to have"; it is the difference between a 10 MB context and a 2 GB context.

`.dockerignore` syntax mirrors `.gitignore`: glob patterns, leading `!` for negation, leading `/` for repo-relative paths. Common entries: `node_modules`, `.git`, `*.log`, `dist/`, build outputs, IDE config, environment files. Many of these are also security-relevant — `.env` files in the context can leak credentials into image layers.

### B. Instructions as Filesystem Transformations

Every Dockerfile instruction either modifies the image filesystem (creating a new layer) or modifies metadata (creating a layer with no filesystem changes, but updating the image config JSON):

| Instruction | Effect on filesystem | Effect on config |
|-------------|----------------------|------------------|
| `FROM` | Initializes from base image's layers | Inherits base config |
| `RUN` | Runs a command, layer = filesystem diff | None |
| `COPY`/`ADD` | Adds files from context (or URL for ADD), layer = added files | None |
| `WORKDIR` | Creates dir if missing | Sets working directory |
| `ENV`, `ARG`, `LABEL`, `EXPOSE`, `USER`, `VOLUME`, `STOPSIGNAL` | None | Updates config field |
| `CMD`, `ENTRYPOINT`, `HEALTHCHECK` | None | Sets runtime defaults |

Each `RUN` produces exactly one layer. So `RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*` is a layer with curl installed and the apt cache removed; splitting it into three `RUN` lines would produce three layers and the apt cache would persist in the middle one (still in the final image, since deletion in a later layer does not free the bytes from earlier layers).

`COPY` always reads from the build context and writes into the image. `ADD` does the same plus two extras: it can fetch a URL (mostly discouraged — no caching guarantees, no checksum verification by default), and it auto-extracts local tar archives. The community consensus: prefer `COPY` for clarity unless you specifically want one of `ADD`'s extras.

`CMD` and `ENTRYPOINT` together define what runs when the container starts. The mental model:

- `ENTRYPOINT` is the program. `CMD` is the default arguments.
- `ENTRYPOINT ["python", "app.py"]` + `CMD ["--port", "8080"]` runs `python app.py --port 8080`. `docker run image --port 9090` overrides `CMD` to `--port 9090` but the entrypoint stays.
- Use the **exec form** `["arg1", "arg2"]` over the **shell form** `arg1 arg2`. Shell form wraps your command in `/bin/sh -c "..."`, which means signals (SIGTERM from `docker stop`) go to the shell, not your process. Your process never receives them and gets SIGKILL'd ten seconds later.

### C. Layer Caching: The Algorithm That Decides Build Time

`docker build` walks instructions in order. For each instruction, it computes a cache key. If the cache key matches an existing layer in the local store, the layer is reused and the instruction is skipped. Otherwise the instruction runs, produces a new layer, and from this point on every subsequent instruction is also a cache miss (the parent layer's digest is part of the next key).

The cache key inputs are:

| Instruction | Cache key includes |
|-------------|---------------------|
| `FROM` | Base image digest |
| `RUN` | Instruction text + parent layer digest |
| `COPY`/`ADD` (local) | Hash of the file contents being copied + parent layer digest |
| `ARG`, `ENV`, etc. | Instruction text + parent layer digest |

Two key implications:

1. **Order from least- to most-changing.** Put your slowest, most-stable instructions first (`apt-get install`, `pip install -r requirements.txt`), then your most-volatile (`COPY . /app`). This way, editing source code only invalidates the last layer, not the dependency installation.
2. **Copy dependency manifests separately.** `COPY package.json package-lock.json ./` followed by `RUN npm ci` is different from `COPY . . && RUN npm ci`. The first invalidates `npm ci` only when those two files change; the second invalidates it on *any* file change.

`docker build --no-cache` disables this entirely (every instruction reruns). `docker build --cache-from <image>` lets CI pull a previous build's layers from a registry and use them as cache, which is how CI builds stay fast despite running on fresh runners.

### D. BuildKit: Graph Execution and Cache Mounts

The classic builder is sequential and stateless. **BuildKit** (default since Docker 23.0; enabled with `DOCKER_BUILDKIT=1` on older versions) is a complete rewrite that:

1. **Parses the Dockerfile into a DAG (directed acyclic graph).** Each instruction is a node; edges are dependencies (instruction N depends on the layer N-1 produced).
2. **Executes independent nodes in parallel.** In a multi-stage Dockerfile with two unrelated build stages, BuildKit runs them concurrently. Output of an unused stage is never built.
3. **Skips unused outputs.** If only the final stage is targeted (`docker build --target prod`), stages that do not feed into `prod` are not executed.
4. **Adds cache mounts.** `RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt` exposes a writable cache directory that survives between builds *outside* the layer system. The downloaded wheels persist; the resulting layer does not contain them.
5. **Adds bind mounts and secret mounts.** `RUN --mount=type=bind,source=.,target=/src ...` and `RUN --mount=type=secret,id=mytoken cat /run/secrets/mytoken` let you read context files or one-time secrets without baking them into a layer.
6. **Adds multi-platform builds.** `docker buildx build --platform linux/amd64,linux/arm64 .` runs the build for each target architecture in parallel, producing a manifest list that the registry serves to the right platform automatically.

The frontend is also pluggable. The default frontend is `dockerfile.v0`, which speaks Dockerfile syntax. Other frontends (`buildpacks`, `Earthly`, custom HCL frontends) emit the same LLB (Low-Level Build) intermediate representation that BuildKit executes, so the entire DAG/cache machinery is reusable.

### E. Multi-Stage Builds: Compile Once, Ship Lean

Many languages have a heavyweight compile step (`go build`, `npm run build`, `mvn package`) producing a small binary or artifact. You do not want the compiler, intermediate object files, or `node_modules` in the final image — they bloat it and broaden the attack surface.

A multi-stage Dockerfile uses multiple `FROM` directives to chain images. A typical pattern:

```dockerfile
# Stage 1: builder
FROM node:20 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: runtime
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
```

The final image contains only the nginx layers plus the small `dist` directory. The `node:20` stage with all its build tooling is discarded — its layers stay in BuildKit's local cache for next time but are never pushed.

You can have any number of stages, name them with `AS <name>`, copy from any earlier stage with `COPY --from=<name>`, and target a specific stage with `docker build --target <name>` to build only up to that point (useful for "test" or "lint" stages that you run in CI but never ship).

### From Theory to the Instructions Below

- `FROM <image>` — sets the base layer digest, which is the root of every cache key downstream.
- `WORKDIR /app` — config-only, no layer with content.
- `COPY package.json /app/` — context tarball lookup, file-content hash drives caching.
- `RUN apt-get install -y ...` — text + parent digest = cache key; the resulting filesystem diff = the layer.
- `ENV NODE_ENV=production` — config metadata; child of every `RUN` after it sees this env var.
- `EXPOSE 8080` — pure documentation in the config; does not actually open a port.
- `CMD ["node", "server.js"]` — config metadata; what `runc` will `execve` if no command is supplied to `docker run`.
- `docker build -t myapp:1.0 .` — pack context (respecting `.dockerignore`), stream to daemon, walk instructions, hit or miss cache for each, push the resulting image into the local store under tag `myapp:1.0`.
- `docker buildx build --platform linux/amd64,linux/arm64 -t myapp:1.0 --push .` — BuildKit fan-out, two parallel architecture builds, registry push of the resulting manifest list.

The remaining sections walk each instruction. Whenever you wonder "is this rebuilding everything?", trace the cache-key inputs above.

---

## 1. What is a Dockerfile?

A Dockerfile is a **configuration file** for creating Docker images. When you write commands in a text file, Docker executes them in order to create an image.

```
Dockerfile → docker build → Docker Image → docker run → Container
(Blueprint)    (Build)       (Template)      (Run)      (Instance)
```

### Why use a Dockerfile?

| Advantage | Description |
|-----------|-------------|
| **Reproducibility** | Create identical images repeatedly |
| **Automation** | No manual setup needed |
| **Version control** | Track history with Git |
| **Documentation** | Environment setup recorded as code |

---

## 2. Dockerfile Basic Syntax

### Basic Structure

```dockerfile
# Comment
INSTRUCTION argument
```

### Main Instructions

| Instruction | Description | Example |
|-------------|-------------|---------|
| `FROM` | Base image | `FROM node:18` |
| `WORKDIR` | Working directory | `WORKDIR /app` |
| `COPY` | Copy files | `COPY . .` |
| `RUN` | Execute command during build | `RUN npm install` |
| `CMD` | Container startup command | `CMD ["npm", "start"]` |
| `EXPOSE` | Expose port | `EXPOSE 3000` |
| `ENV` | Environment variable | `ENV NODE_ENV=production` |

---

## 3. Instruction Details

### FROM - Base Image

Every Dockerfile starts with `FROM`.

```dockerfile
# Basic
FROM ubuntu:22.04

# Node.js image
FROM node:18

# Alpine: ~175 MB vs ~1 GB full image — smaller attack surface and faster CI pulls
FROM node:18-alpine

# Multi-stage build — build tools stay in 'builder', excluded from final image
FROM node:18 AS builder
FROM nginx:alpine AS production
```

### WORKDIR - Working Directory

Sets the directory where subsequent commands will execute.

```dockerfile
WORKDIR /app

# Subsequent commands execute in /app
COPY . .          # Copy to /app
RUN npm install   # Execute in /app
```

### COPY - Copy Files

Copies files from host to image.

```dockerfile
# Copy file
COPY package.json .

# Copy directory
COPY src/ ./src/

# Copy all files
COPY . .

# Copy multiple files
COPY package.json package-lock.json ./
```

### ADD vs COPY

```dockerfile
# COPY: Simple copy (recommended)
COPY local-file.txt /app/

# ADD: URL download, archive extraction
ADD https://example.com/file.tar.gz /app/
ADD archive.tar.gz /app/  # Auto-extracts
```

### RUN - Execute Build Command

Executes during image build.

```dockerfile
# Basic
RUN npm install

# Combine in one RUN so the apt cache never persists in a committed layer
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*  # Remove apt cache; must be in same RUN to avoid bloating the image

# Layer caching: copy dependency manifest first (changes rarely), then install, then copy source (changes often)
COPY package*.json ./   # Dependency manifest only — changes less often than source code
RUN npm install         # Cached as long as package*.json is unchanged
COPY . .                # Source changes every build; placed last to preserve npm install cache
```

### CMD - Container Startup Command

Executes when container starts.

```dockerfile
# exec form (recommended) — no shell wrapper, so the process receives
# OS signals (e.g., SIGTERM) directly for graceful shutdown
CMD ["npm", "start"]
CMD ["node", "app.js"]

# shell form — runs via /bin/sh -c; process won't receive signals directly
CMD npm start
```

### ENTRYPOINT vs CMD

```dockerfile
# ENTRYPOINT = fixed command, CMD = overridable default argument
ENTRYPOINT ["node"]
CMD ["app.js"]           # Default arg; override with: docker run myimage other.js
# Executes: node app.js

# docker run myimage other.js
# Executes: node other.js (ENTRYPOINT stays, only CMD is replaced)
```

### ENV - Environment Variables

```dockerfile
# Single variable
ENV NODE_ENV=production

# Multiple variables
ENV NODE_ENV=production \
    PORT=3000 \
    DB_HOST=localhost
```

### EXPOSE - Document Port

```dockerfile
# EXPOSE is documentation only — does not actually publish the port (use -p at runtime for that)
EXPOSE 3000
EXPOSE 80 443
```

### ARG - Build-time Variables

```dockerfile
# ARG: available only at build time — use for values that should not persist in the running container
ARG NODE_VERSION=18
FROM node:${NODE_VERSION}

# Promote ARG to ENV so the value is available at runtime too (e.g., for version endpoints)
ARG APP_VERSION=1.0.0
ENV APP_VERSION=${APP_VERSION}
```

```bash
# Pass value during build
docker build --build-arg NODE_VERSION=20 .
```

---

## 4. Practice Examples

### Example 1: Node.js Application

**Project structure:**
```
my-node-app/
├── Dockerfile
├── package.json
└── app.js
```

**package.json:**
```json
{
  "name": "my-node-app",
  "version": "1.0.0",
  "main": "app.js",
  "scripts": {
    "start": "node app.js"
  },
  "dependencies": {
    "express": "^4.18.2"
  }
}
```

**app.js:**
```javascript
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
  res.json({ message: 'Hello from Docker!', version: '1.0.0' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

**Dockerfile:**
```dockerfile
# Base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy dependency manifest first — changes less often than source code
COPY package*.json ./

# Install deps — this layer is cached as long as package*.json hasn't changed
RUN npm install

# Copy source last — changes every build, so it doesn't invalidate npm install cache
COPY . .

# Document the port this app listens on (actual mapping done with -p at runtime)
EXPOSE 3000

# exec form: process receives OS signals directly (needed for graceful shutdown)
CMD ["npm", "start"]
```

**Build and run:**
```bash
# Build image
docker build -t my-node-app .

# Run container
docker run -d -p 3000:3000 --name node-app my-node-app

# Test
curl http://localhost:3000

# Cleanup
docker rm -f node-app
```

### Example 2: Python Flask Application

**Project structure:**
```
my-flask-app/
├── Dockerfile
├── requirements.txt
└── app.py
```

**requirements.txt:**
```
flask==3.0.0
gunicorn==21.2.0
```

**app.py:**
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify(message='Hello from Flask in Docker!')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # --no-cache-dir: skip storing downloaded packages in the layer

# Copy source
COPY . .

EXPOSE 5000

# Run with Gunicorn (production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

**Build and run:**
```bash
docker build -t my-flask-app .
docker run -d -p 5000:5000 my-flask-app
curl http://localhost:5000
```

### Example 3: Static Website (Nginx)

**Project structure:**
```
my-website/
├── Dockerfile
├── nginx.conf
└── public/
    └── index.html
```

**public/index.html:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>My Docker Website</title>
</head>
<body>
    <h1>Hello from Nginx in Docker!</h1>
</body>
</html>
```

**Dockerfile:**
```dockerfile
# Alpine: ~5 MB base — ideal for serving static files with minimal overhead
FROM nginx:alpine

# Copy custom config (optional)
# COPY nginx.conf /etc/nginx/nginx.conf

# Copy static files
COPY public/ /usr/share/nginx/html/

EXPOSE 80

# "daemon off;" keeps nginx in the foreground so Docker can track the process as PID 1
CMD ["nginx", "-g", "daemon off;"]
```

---

## 5. Multi-stage Build

Separate build and runtime environments to reduce image size.

### React App Example

```dockerfile
# Stage 1: Build — node_modules + build toolchain (~300 MB) are discarded after this stage
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Runtime — only the static build output is copied; final image is ~25 MB
FROM nginx:alpine

# --from=builder: pull artifacts from the build stage without carrying over node_modules
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Go App Example

```dockerfile
# Stage 1: Build — Go compiler + stdlib needed only at compile time
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .
# Static binary: no external C library deps, so the runtime stage needs almost nothing
RUN go build -o main .

# Stage 2: Runtime — alpine:latest is ~5 MB; use 'scratch' for even smaller (~0 MB base)
FROM alpine:latest

WORKDIR /app
COPY --from=builder /app/main .

EXPOSE 8080
CMD ["./main"]
```

**Size comparison:**
```
golang:1.21-alpine  →  ~300MB (build environment)
Final image         →  ~15MB (runtime environment)
```

---

## 6. Best Practices

### .dockerignore File

Exclude unnecessary files from build.

```
# .dockerignore — reduces build context size and prevents secrets/large dirs from leaking into the image
node_modules
npm-debug.log
.git
.gitignore
.env
*.md
Dockerfile
.dockerignore
```

### Layer Optimization

```dockerfile
# Bad: Copying everything first means ANY source change invalidates the npm install cache
COPY . .
RUN npm install

# Good: Copy manifest first — npm install is cached until package.json changes
COPY package*.json ./
RUN npm install
COPY . .   # Source changes don't trigger a reinstall
```

### Use Small Images

```dockerfile
# Large — full Debian with build tools; only needed if you compile native addons
FROM node:18           # ~1GB

# Recommended — Alpine Linux: ~5 MB base, minimal packages, smaller attack surface
FROM node:18-alpine    # ~175MB

# Minimal — Debian slim: smaller than full but includes glibc (better native addon compat than Alpine)
FROM node:18-slim      # ~200MB
```

### Security

```dockerfile
# Run as non-root user (limits damage if container is compromised)
FROM node:18-alpine

# -S = system account (no home dir, no login shell) — appropriate for service processes
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

WORKDIR /app
COPY --chown=appuser:appgroup . .  # --chown ensures the non-root user can read the copied files
```

---

## 7. Image Build Commands

```bash
# Basic build
docker build -t imagename .

# Specify tag
docker build -t myapp:1.0 .

# Use different Dockerfile
docker build -f Dockerfile.prod -t myapp:prod .

# Pass build arguments
docker build --build-arg NODE_ENV=production -t myapp .

# --no-cache: force rebuild all layers — useful when a base image or remote dep changed
docker build --no-cache -t myapp .

# --progress=plain: show full build output — easier to debug failed RUN steps
docker build --progress=plain -t myapp .
```

---

## Command Summary

| Dockerfile Instruction | Description |
|------------------------|-------------|
| `FROM` | Specify base image |
| `WORKDIR` | Set working directory |
| `COPY` | Copy files/directories |
| `RUN` | Execute command during build |
| `CMD` | Container startup command |
| `EXPOSE` | Document port |
| `ENV` | Set environment variable |
| `ARG` | Build-time variable |
| `ENTRYPOINT` | Fixed execution command |

---

## Exercises

### Exercise 1: Write Your First Dockerfile

Create a Dockerfile for a simple Python Flask application.

1. Create a project directory and add these files:
   - `requirements.txt` containing `flask==3.0.0`
   - `app.py` with a Flask app that returns `{"message": "Hello, Docker!"}` on the root route
2. Write a `Dockerfile` using `python:3.11-slim` as the base image, a non-root user, and proper layer caching (copy `requirements.txt` before `app.py`)
3. Build the image: `docker build -t flask-hello:1.0 .`
4. Run the container on port 5000: `docker run -d -p 5000:5000 flask-hello:1.0`
5. Test with `curl http://localhost:5000` and confirm the response

### Exercise 2: Layer Caching Experiment

Observe how layer caching affects build times.

1. Start with a Node.js Dockerfile that copies everything first and then runs `npm install`:
   ```dockerfile
   FROM node:18-alpine
   WORKDIR /app
   COPY . .
   RUN npm install
   CMD ["node", "app.js"]
   ```
2. Build it (`docker build -t cache-test:bad .`) and note the build time
3. Rewrite the Dockerfile to copy `package*.json` first, then run `npm install`, then copy the rest
4. Build again (`docker build -t cache-test:good .`) and note the build time
5. Modify only `app.js`, rebuild both versions, and compare how much of the build is cached in each case

### Exercise 3: Multi-Stage Build

Reduce image size using a multi-stage build.

1. Create a simple Go program (`main.go`) that prints "Hello from Go!"
2. Write a single-stage Dockerfile using `golang:1.21-alpine` and build it; record the image size
3. Rewrite with a multi-stage build: compile in `golang:1.21-alpine` and copy only the binary to `FROM scratch` or `alpine:latest` for the final stage
4. Compare the sizes of the single-stage and multi-stage images with `docker images`
5. Verify the multi-stage image runs correctly

### Exercise 4: CMD vs ENTRYPOINT

Understand the difference between `CMD` and `ENTRYPOINT` through experimentation.

1. Create a Dockerfile with `ENTRYPOINT ["echo"]` and `CMD ["Hello, World!"]`
2. Build and run it to see the default output
3. Override CMD at runtime: `docker run <image> "Goodbye, World!"` — what happens?
4. Try to override ENTRYPOINT: `docker run --entrypoint /bin/sh <image>` — how does this differ?
5. Modify the Dockerfile to use only `CMD ["echo", "Hello, World!"]` (without ENTRYPOINT), rebuild, and try the same overrides. Document the differences.

### Exercise 5: .dockerignore and Build Context

Optimize the build context using `.dockerignore`.

1. Create a project with `node_modules/`, `.git/`, `.env`, and source files
2. Build without a `.dockerignore` and run `docker build --no-cache --progress=plain -t context-test .` to observe the build context size in the output
3. Create a `.dockerignore` file excluding `node_modules`, `.git`, `.env`, and `*.log`
4. Rebuild and compare the build context size
5. Run `docker build --no-cache --progress=plain -t context-test:optimized .` and verify the context is smaller

---

**Previous**: [Docker Images and Containers](./02_Images_and_Containers.md) | **Next**: [Docker Compose](./04_Docker_Compose.md)
