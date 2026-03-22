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

While pulling pre-built images from Docker Hub is convenient, real-world projects require custom images tailored to your specific application and dependencies. The Dockerfile is the standard mechanism for defining these custom images as code. By learning Dockerfile syntax and best practices such as multi-stage builds and layer optimization, you gain full control over your application's packaging and can ensure consistent, secure, and efficient container images.

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
