#!/bin/bash
# Exercises for Lesson 03: Dockerfile
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Write Your First Dockerfile ===
# Problem: Create a Dockerfile for a simple Python Flask application with
# proper layer caching and a non-root user.
exercise_1() {
    echo "=== Exercise 1: Write Your First Dockerfile ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- requirements.txt ---"
    cat << 'SOLUTION'
flask==3.0.0
SOLUTION
    echo ""
    echo "--- app.py ---"
    cat << 'SOLUTION'
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify(message="Hello, Docker!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
SOLUTION
    echo ""
    echo "--- Dockerfile ---"
    cat << 'SOLUTION'
# Use slim variant — ~155 MB vs ~920 MB full image; smaller attack surface
FROM python:3.11-slim

# Set working directory before any file operations
WORKDIR /app

# Create a non-root user for security — limits damage if the container is compromised
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy dependency manifest first — changes less often than source code,
# so Docker caches the pip install layer across builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir: skip caching downloaded packages in the layer (saves ~30-50 MB)

# Copy application source last — source changes every build,
# placed after pip install to preserve the dependency cache
COPY app.py .

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user — all subsequent commands run as appuser
USER appuser

# Document the port this app listens on (actual mapping done with -p at runtime)
EXPOSE 5000

# Exec form: process runs as PID 1, receives SIGTERM for graceful shutdown
CMD ["python", "app.py"]
SOLUTION
    echo ""
    echo "--- Build and Run ---"
    cat << 'SOLUTION'
# Build the image with a tag and version
docker build -t flask-hello:1.0 .

# Run the container, mapping host port 5000 to container port 5000
docker run -d -p 5000:5000 flask-hello:1.0

# Test the endpoint
curl http://localhost:5000
# {"message":"Hello, Docker!"}

# Cleanup
docker rm -f $(docker ps -q --filter ancestor=flask-hello:1.0)
SOLUTION
}

# === Exercise 2: Layer Caching Experiment ===
# Problem: Observe how layer caching affects build times by comparing
# two Dockerfile strategies.
exercise_2() {
    echo "=== Exercise 2: Layer Caching Experiment ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- Bad Dockerfile (no caching optimization) ---"
    cat << 'SOLUTION'
# BAD: Copies everything first, then installs dependencies
# Any change to ANY file (even a comment in app.js) invalidates
# the npm install cache, forcing a full reinstall every build.
FROM node:18-alpine
WORKDIR /app
COPY . .
RUN npm install
CMD ["node", "app.js"]
SOLUTION
    echo ""
    echo "--- Good Dockerfile (optimized caching) ---"
    cat << 'SOLUTION'
# GOOD: Copy dependency manifest first, install, then copy source
# npm install is cached as long as package*.json hasn't changed.
# Source code changes (app.js) only invalidate the COPY . . layer.
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
CMD ["node", "app.js"]
SOLUTION
    echo ""
    echo "--- Experiment Steps ---"
    cat << 'SOLUTION'
# Build the "bad" version
time docker build -t cache-test:bad -f Dockerfile.bad .
# Note: full npm install runs every time

# Build the "good" version
time docker build -t cache-test:good -f Dockerfile.good .
# Note: full npm install runs on first build

# Now modify ONLY app.js (not package.json)
echo "// small change" >> app.js

# Rebuild both:
time docker build -t cache-test:bad -f Dockerfile.bad .
# BAD: npm install runs AGAIN because COPY . . changed before it
# Typical time: 15-30 seconds (downloads all deps)

time docker build -t cache-test:good -f Dockerfile.good .
# GOOD: npm install is CACHED (package.json unchanged)
# Only the COPY . . layer rebuilds
# Typical time: 1-3 seconds

# Key takeaway: Always copy dependency manifests before source code.
# This is the single most impactful Dockerfile optimization.
SOLUTION
}

# === Exercise 3: Multi-Stage Build ===
# Problem: Reduce image size using multi-stage builds with Go.
exercise_3() {
    echo "=== Exercise 3: Multi-Stage Build ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- main.go ---"
    cat << 'SOLUTION'
package main

import "fmt"

func main() {
    fmt.Println("Hello from Go!")
}
SOLUTION
    echo ""
    echo "--- Dockerfile.single (single-stage) ---"
    cat << 'SOLUTION'
# Single-stage: the Go compiler and stdlib stay in the final image
FROM golang:1.21-alpine
WORKDIR /app
COPY main.go .
RUN go build -o main .
CMD ["./main"]
# Image size: ~300 MB (includes Go compiler, stdlib, apk packages)
SOLUTION
    echo ""
    echo "--- Dockerfile.multi (multi-stage) ---"
    cat << 'SOLUTION'
# Stage 1: Build — Go compiler needed only at compile time
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY main.go .
# CGO_ENABLED=0 produces a fully static binary — no libc dependency,
# so the binary runs on 'scratch' (empty image) or alpine without issues
RUN CGO_ENABLED=0 go build -o main .

# Stage 2: Runtime — only the compiled binary is copied
# 'scratch' is an empty image (~0 MB base) — the absolute minimum
FROM scratch
COPY --from=builder /app/main /main
CMD ["/main"]
# Final image size: ~2 MB (just the Go binary)
SOLUTION
    echo ""
    echo "--- Size Comparison ---"
    cat << 'SOLUTION'
# Build both images
docker build -t go-single -f Dockerfile.single .
docker build -t go-multi -f Dockerfile.multi .

# Compare sizes
docker images | grep go-
# REPOSITORY   TAG     SIZE
# go-single    latest  ~300MB
# go-multi     latest  ~2MB

# The multi-stage image is ~150x smaller!
# Benefits:
# - Faster pulls and deploys (less data to transfer)
# - Smaller attack surface (no shell, no package manager, no compiler)
# - Lower storage costs in registries

# Verify it still runs correctly
docker run --rm go-multi
# Output: Hello from Go!
SOLUTION
}

# === Exercise 4: CMD vs ENTRYPOINT ===
# Problem: Understand the difference through experimentation.
exercise_4() {
    echo "=== Exercise 4: CMD vs ENTRYPOINT ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- Dockerfile with ENTRYPOINT + CMD ---"
    cat << 'SOLUTION'
FROM alpine
ENTRYPOINT ["echo"]
CMD ["Hello, World!"]
# ENTRYPOINT = fixed command (always runs 'echo')
# CMD = default argument (can be overridden at runtime)
SOLUTION
    echo ""
    echo "--- Experiment Steps ---"
    cat << 'SOLUTION'
# Build the image
docker build -t ep-test .

# Run with default CMD
docker run --rm ep-test
# Output: Hello, World!
# Executes: echo "Hello, World!"

# Override CMD at runtime
docker run --rm ep-test "Goodbye, World!"
# Output: Goodbye, World!
# Executes: echo "Goodbye, World!"
# ENTRYPOINT (echo) stays fixed; only CMD is replaced

# Override ENTRYPOINT entirely
docker run --rm --entrypoint /bin/sh ep-test -c "ls /"
# This replaces BOTH entrypoint and cmd
# Useful for debugging: get a shell inside a container whose
# entrypoint normally runs an application

# --- Now test with CMD only (no ENTRYPOINT) ---
# Dockerfile.cmd-only:
#   FROM alpine
#   CMD ["echo", "Hello, World!"]

# Build and run
docker build -t cmd-test -f Dockerfile.cmd-only .

docker run --rm cmd-test
# Output: Hello, World!

docker run --rm cmd-test ls /
# Output: (directory listing)
# The ENTIRE CMD is replaced — "echo Hello World" is gone
# This is the key difference:
#   ENTRYPOINT+CMD: only the argument changes
#   CMD only: the whole command changes

# Summary:
# ┌─────────────┬──────────────────────────────────────────┐
# │ Pattern     │ Behavior                                  │
# ├─────────────┼──────────────────────────────────────────┤
# │ ENTRYPOINT  │ Fixed command, always runs                │
# │ CMD         │ Default args, easily overridden           │
# │ Both        │ ENTRYPOINT = command, CMD = default args  │
# │ CMD only    │ Entire command is replaceable             │
# └─────────────┴──────────────────────────────────────────┘
SOLUTION
}

# === Exercise 5: .dockerignore and Build Context ===
# Problem: Optimize build context using .dockerignore.
exercise_5() {
    echo "=== Exercise 5: .dockerignore and Build Context ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Create a project with large/unnecessary directories
mkdir -p myproject/node_modules/.cache myproject/.git/objects
echo "SECRET_KEY=abc123" > myproject/.env
dd if=/dev/zero of=myproject/node_modules/.cache/big_file bs=1M count=50 2>/dev/null
echo "console.log('hello');" > myproject/app.js
echo '{"dependencies":{}}' > myproject/package.json
cat > myproject/Dockerfile << 'EOF'
FROM node:18-alpine
WORKDIR /app
COPY . .
CMD ["node", "app.js"]
EOF

# Step 2: Build WITHOUT .dockerignore — observe context size
cd myproject
docker build --no-cache --progress=plain -t context-test . 2>&1 | head -5
# => transferring context: 52.4MB
# The 50 MB fake file in node_modules is sent to the Docker daemon!
# .git and .env are also included — security risk

# Step 3: Create .dockerignore
cat > .dockerignore << 'EOF'
node_modules
.git
.env
*.log
*.md
Dockerfile
.dockerignore
EOF

# Step 4: Rebuild WITH .dockerignore
docker build --no-cache --progress=plain -t context-test:optimized . 2>&1 | head -5
# => transferring context: 4.1KB
# Massive reduction! node_modules, .git, .env are all excluded

# Key benefits of .dockerignore:
# 1. Faster builds — less data transferred to Docker daemon
# 2. Smaller images — excluded files can't accidentally end up in layers
# 3. Security — .env files and .git history stay out of the image
# 4. Cache stability — changes to ignored files don't invalidate layers
SOLUTION
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
echo ""
echo "All exercises completed!"
