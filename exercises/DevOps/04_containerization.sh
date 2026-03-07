#!/bin/bash
# Exercises for Lesson 04: Containerization
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Multi-Stage Dockerfile ===
# Problem: Write an optimized multi-stage Dockerfile for a Python Flask
# application that minimizes image size and follows security best practices.
exercise_1() {
    echo "=== Exercise 1: Multi-Stage Dockerfile ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Dockerfile — Multi-stage build for a Python Flask application
# Target: <100MB final image, non-root user, no build tools in prod

# Stage 1: Build dependencies in a full image
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies (gcc needed for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image (minimal)
FROM python:3.12-slim AS production

# Security: create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy only the virtual environment from builder (no gcc, no build tools)
COPY --from=builder /opt/venv /opt/venv

# Install only runtime libraries (no -dev packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --chown=appuser:appuser . .

# Use the virtual environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Health check for orchestrator integration
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Switch to non-root user
USER appuser

EXPOSE 5000

# Use exec form for proper signal handling (PID 1)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]

# Image size comparison:
#   python:3.12          ~1GB
#   python:3.12-slim     ~150MB
#   This multi-stage      ~90MB (no build tools, no pip cache)
SOLUTION
}

# === Exercise 2: Docker Compose for Local Development ===
# Problem: Write a docker-compose.yml for a 3-tier application
# (web, API, database) with health checks and named volumes.
exercise_2() {
    echo "=== Exercise 2: Docker Compose for Local Development ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# docker-compose.yml
version: "3.9"

services:
  # --- PostgreSQL Database ---
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${DB_USER:-appuser}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-secret}
      POSTGRES_DB: ${DB_NAME:-appdb}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # --- Redis Cache ---
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    restart: unless-stopped

  # --- API Server ---
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
      target: development    # Use dev stage with hot-reload
    environment:
      DATABASE_URL: postgresql://${DB_USER:-appuser}:${DB_PASSWORD:-secret}@db:5432/${DB_NAME:-appdb}
      REDIS_URL: redis://redis:6379/0
      FLASK_ENV: development
    volumes:
      - ./api:/app           # Hot-reload: mount source code
      - /app/__pycache__     # Exclude pycache from mount
    ports:
      - "5000:5000"
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 15s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  # --- Frontend (Nginx) ---
  web:
    image: nginx:1.25-alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./frontend/dist:/usr/share/nginx/html:ro
    ports:
      - "80:80"
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped

volumes:
  pgdata:
    name: myapp-pgdata

# Key design decisions:
# 1. depends_on with condition: service_healthy — waits for real readiness
# 2. Named volume (pgdata) — data survives container recreation
# 3. Source mount for API — enables hot-reload during development
# 4. Alpine images — smaller footprint for local dev
# 5. .env file support — ${VAR:-default} pattern
SOLUTION
}

# === Exercise 3: Container Image Optimization ===
# Problem: Analyze a Docker image and reduce its size by 60%.
exercise_3() {
    echo "=== Exercise 3: Container Image Optimization ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Optimization techniques and their impact:

optimization_techniques = [
    {
        "technique": "Use slim/alpine base image",
        "before": "python:3.12 (1.0 GB)",
        "after": "python:3.12-slim (150 MB)",
        "saving_pct": 85,
        "command": "FROM python:3.12-slim",
    },
    {
        "technique": "Multi-stage build (remove build tools)",
        "before": "150 MB (with gcc, pip cache)",
        "after": "90 MB (runtime only)",
        "saving_pct": 40,
        "command": "COPY --from=builder /opt/venv /opt/venv",
    },
    {
        "technique": "Combine RUN layers",
        "before": "5 layers x apt-get (each cached separately)",
        "after": "1 layer with cleanup",
        "saving_pct": 15,
        "command": "RUN apt-get update && apt-get install -y pkg && rm -rf /var/lib/apt/lists/*",
    },
    {
        "technique": "Use .dockerignore",
        "before": "Copies .git, node_modules, __pycache__, .env",
        "after": "Only copies needed files",
        "saving_pct": 10,
        "command": "# .dockerignore\n.git\nnode_modules\n__pycache__\n*.pyc\n.env",
    },
    {
        "technique": "pip --no-cache-dir",
        "before": "Pip cache stored in image (~50MB)",
        "after": "No cache in image",
        "saving_pct": 5,
        "command": "RUN pip install --no-cache-dir -r requirements.txt",
    },
]

for opt in optimization_techniques:
    print(f"  {opt['technique']} (-{opt['saving_pct']}%)")
    print(f"    Before: {opt['before']}")
    print(f"    After:  {opt['after']}")
    print()

# Commands to analyze image size:
# docker images myapp                    # Total size
# docker history myapp                   # Size per layer
# docker run --rm wagoodman/dive myapp   # Interactive layer explorer
SOLUTION
}

# === Exercise 4: Container Security Scanning ===
# Problem: Set up container image scanning in CI and interpret the results.
exercise_4() {
    echo "=== Exercise 4: Container Security Scanning ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# GitHub Actions step for Trivy container scanning
# Add this to your CI workflow after docker build:

# .github/workflows/ci.yml (scanning job)
# security-scan:
#   runs-on: ubuntu-latest
#   steps:
#     - uses: actions/checkout@v4
#     - name: Build image
#       run: docker build -t myapp:scan .
#     - name: Run Trivy vulnerability scanner
#       uses: aquasecurity/trivy-action@master
#       with:
#         image-ref: myapp:scan
#         format: table
#         exit-code: 1              # Fail CI on HIGH/CRITICAL
#         severity: HIGH,CRITICAL
#         ignore-unfixed: true      # Skip vulns with no fix available

# Interpreting scan results:
severity_actions = {
    "CRITICAL": "Block deployment. Fix immediately. Upgrade base image or package.",
    "HIGH":     "Block deployment. Fix within 24 hours.",
    "MEDIUM":   "Track in backlog. Fix within 1 sprint.",
    "LOW":      "Acknowledge. Fix during dependency updates.",
}

print("Vulnerability Severity Response Plan:")
for severity, action in severity_actions.items():
    print(f"  {severity:10s}: {action}")

# Best practices for secure containers:
print("\nContainer Security Checklist:")
checklist = [
    "Run as non-root user (USER directive)",
    "Use read-only filesystem (--read-only flag)",
    "Drop all capabilities (--cap-drop ALL)",
    "Pin base image digests (FROM image@sha256:...)",
    "Scan images in CI before pushing to registry",
    "Sign images with cosign/notation",
    "Use minimal base images (distroless, alpine)",
    "Never store secrets in image layers",
]
for item in checklist:
    print(f"  [x] {item}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 04: Containerization"
echo "==================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
