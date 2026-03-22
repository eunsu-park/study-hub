#!/bin/bash
# Exercises for Lesson 04: Docker Compose
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Two-Service Stack ===
# Problem: Create a Compose stack with a Python web app and Redis.
exercise_1() {
    echo "=== Exercise 1: Two-Service Stack ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- docker-compose.yml ---"
    cat << 'SOLUTION'
services:
  redis:
    image: redis:7-alpine
    # Alpine variant: ~30 MB vs ~130 MB full image — smaller and faster to pull

  web:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    environment:
      # 'redis' hostname resolves via Compose's embedded DNS to the redis container's IP
      - DATABASE_URL=redis://redis:6379
    depends_on:
      - redis
      # depends_on ensures redis starts first, but does NOT wait for it to be "ready"
      # For production, use healthcheck + condition: service_healthy
    command: python -c "import http.server; http.server.HTTPServer(('',5000), http.server.SimpleHTTPRequestHandler).serve_forever()"
    # Simple HTTP server as placeholder — replace with your actual app command
SOLUTION
    echo ""
    echo "--- Commands ---"
    cat << 'SOLUTION'
# Start both services in detached mode
docker compose up -d

# Verify both services are running
docker compose ps
# NAME                SERVICE   STATUS    PORTS
# project-redis-1     redis     running   6379/tcp
# project-web-1       web       running   0.0.0.0:5000->5000/tcp

# View logs for the redis service only
docker compose logs redis
# Shows Redis startup messages: "Ready to accept connections"

# Execute a Redis CLI command inside the redis container
docker compose exec redis redis-cli ping
# Output: PONG
# This confirms Redis is running and accepting connections

# Tear down everything
docker compose down
# Stops and removes containers, the default network, but NOT volumes

# Confirm all containers are removed
docker ps -a
# No containers from this project should remain
SOLUTION
}

# === Exercise 2: Persistent Database with Health Check ===
# Problem: PostgreSQL with health check and dependent app startup.
exercise_2() {
    echo "=== Exercise 2: Persistent Database with Health Check ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- docker-compose.yml ---"
    cat << 'SOLUTION'
services:
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=appuser
      - POSTGRES_PASSWORD=apppass
      - POSTGRES_DB=appdb
    volumes:
      - pgdata:/var/lib/postgresql/data
      # Named volume: data persists across container removal
      # Managed by Docker at /var/lib/docker/volumes/pgdata/_data
    healthcheck:
      # pg_isready checks if Postgres is ACCEPTING CONNECTIONS, not just alive
      # This is critical: the process starts ~5s before it accepts queries
      test: ["CMD-SHELL", "pg_isready -U appuser -d appdb"]
      interval: 5s       # Probe every 5 seconds
      timeout: 5s        # Fail the probe if no response in 5s
      retries: 5         # Mark unhealthy after 5 consecutive failures
      start_period: 10s  # Grace period: failures during this window don't count

  app:
    image: alpine:latest
    depends_on:
      db:
        condition: service_healthy
        # Key difference from plain depends_on:
        # - Plain: starts app as soon as db container is CREATED
        # - service_healthy: waits until db passes its healthcheck
        # This prevents "connection refused" errors on app startup
    command: sh -c "echo 'App started — DB is healthy!' && sleep infinity"

volumes:
  pgdata:
SOLUTION
    echo ""
    echo "--- Commands ---"
    cat << 'SOLUTION'
# Start the stack
docker compose up -d

# Observe startup order: db starts first, app waits for health check
docker compose ps
# db: running (healthy)
# app: running  (only started after db became healthy)

# Verify data persistence:
# 1. Create a table in the database
docker compose exec db psql -U appuser -d appdb -c "CREATE TABLE test (id int);"
docker compose exec db psql -U appuser -d appdb -c "INSERT INTO test VALUES (42);"

# 2. Stop and restart the stack (without -v)
docker compose down
docker compose up -d

# 3. Verify data survived the restart
docker compose exec db psql -U appuser -d appdb -c "SELECT * FROM test;"
#  id
# ----
#  42
# Data persists because the named volume 'pgdata' was retained

# 4. Now destroy everything including volumes
docker compose down -v
# The -v flag removes named volumes too
# pgdata is deleted — all database data is gone

# Verify: restarting would create a fresh, empty database
docker compose up -d
docker compose exec db psql -U appuser -d appdb -c "SELECT * FROM test;" 2>&1
# ERROR: relation "test" does not exist
# Volume was destroyed, so all data is gone
docker compose down -v
SOLUTION
}

# === Exercise 3: Development vs Production Environments ===
# Problem: Use multiple Compose files for environment-specific configuration.
exercise_3() {
    echo "=== Exercise 3: Development vs Production Environments ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- docker-compose.yml (base) ---"
    cat << 'SOLUTION'
# Base configuration shared by all environments
services:
  web:
    build: .
    # No ports, env, or volumes here — those are environment-specific
SOLUTION
    echo ""
    echo "--- docker-compose.override.yml (development, auto-merged) ---"
    cat << 'SOLUTION'
# This file is automatically merged when you run 'docker compose up'
# (Compose merges docker-compose.yml + docker-compose.override.yml by default)
services:
  web:
    volumes:
      - .:/app                    # Bind mount source code for live-reload
    environment:
      - NODE_ENV=development      # Enable verbose logging, hot-reload, etc.
    ports:
      - "3001:3000"               # Non-standard port to avoid conflicts
    command: npm run dev           # File-watching dev server instead of production start
SOLUTION
    echo ""
    echo "--- docker-compose.prod.yml (production) ---"
    cat << 'SOLUTION'
services:
  web:
    restart: always               # Auto-restart on crash AND after host reboot
    environment:
      - NODE_ENV=production       # Optimized builds, minimal logging
    ports:
      - "80:3000"                 # Standard HTTP port for production traffic
    # No volume mounts — the image contains the built application
    # No command override — uses the Dockerfile's CMD
SOLUTION
    echo ""
    echo "--- Usage ---"
    cat << 'SOLUTION'
# Development mode (auto-merges override file)
docker compose up
# Merges: docker-compose.yml + docker-compose.override.yml
# Result: port 3001, NODE_ENV=development, source code mounted, npm run dev

# Production mode (explicit file selection)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
# Merges: docker-compose.yml + docker-compose.prod.yml
# Override file is NOT included because we specified files explicitly
# Result: port 80, NODE_ENV=production, restart: always, Dockerfile CMD

# Verify differences:
docker compose ps                    # Dev: port 3001
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps  # Prod: port 80

# Key insight: The override file is a Compose convention, not just a feature.
# Team members get consistent dev environments by default (just 'docker compose up')
# while production configs are opt-in (require explicit -f flags).
SOLUTION
}

# === Exercise 4: Service Scaling ===
# Problem: Scale a service and observe load distribution.
exercise_4() {
    echo "=== Exercise 4: Service Scaling ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- docker-compose.yml ---"
    cat << 'SOLUTION'
services:
  api:
    image: hashicorp/http-echo
    command: ["-text", "Hello from this replica"]
    # Do NOT specify a fixed host port when scaling —
    # multiple containers cannot bind to the same host port.
    # Use a reverse proxy (nginx/traefik) in front instead.
    expose:
      - "5678"    # Internal port only, no host mapping
SOLUTION
    echo ""
    echo "--- Commands ---"
    cat << 'SOLUTION'
# Start with 1 replica (default)
docker compose up -d

# Scale the api service to 3 replicas
docker compose up -d --scale api=3
# Compose creates 2 additional containers for the api service

# Verify three containers are running
docker compose ps
# NAME             SERVICE   STATUS    PORTS
# project-api-1    api       running   5678/tcp
# project-api-2    api       running   5678/tcp
# project-api-3    api       running   5678/tcp

# View logs from all replicas
docker compose logs api
# Each log line is prefixed with the container name (api-1, api-2, api-3)
# This makes it easy to trace which replica handled each request

# Scale back down to 1 replica
docker compose up -d --scale api=1
# Compose stops and removes 2 of the 3 containers

# Verify
docker compose ps
# Only api-1 remains

# Note: For production-grade scaling with load balancing,
# use Kubernetes or Docker Swarm. Compose scaling is mainly
# useful for local development and testing scenarios.

docker compose down
SOLUTION
}

# === Exercise 5: Full-Stack Application Compose ===
# Problem: Three-service stack with network isolation.
exercise_5() {
    echo "=== Exercise 5: Full-Stack Application Compose ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- docker-compose.yml ---"
    cat << 'SOLUTION'
services:
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=${DB_PASSWORD:-secretpass}
    volumes:
      - pgdata:/var/lib/postgresql/data    # Named volume for data persistence
    networks:
      - backend-net                         # Only accessible from backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d myapp"]
      interval: 5s
      timeout: 5s
      retries: 5

  backend:
    image: node:18-alpine
    command: >
      sh -c "echo 'Backend running on port 3000' &&
             node -e \"require('http').createServer((req,res)=>{
               res.end('Hello from backend')
             }).listen(3000)\""
    environment:
      - DB_HOST=db              # Compose DNS resolves 'db' to its container IP
      - DB_PORT=5432
      - DB_NAME=myapp
    depends_on:
      db:
        condition: service_healthy
    networks:
      - frontend-net             # Reachable by frontend
      - backend-net              # Can reach db — acts as gateway between networks
    expose:
      - "3000"

  frontend:
    image: nginx:alpine
    ports:
      - "80:80"                  # Only the frontend is exposed to the host
    depends_on:
      - backend
    networks:
      - frontend-net             # Can reach backend, but NOT db directly
    # Frontend can only talk to backend, not directly to db.
    # This is defense-in-depth: even if the frontend is compromised,
    # the attacker cannot reach the database.

networks:
  frontend-net:
    driver: bridge
    # frontend + backend share this network
  backend-net:
    driver: bridge
    # backend + db share this network
    # frontend is NOT on this network — no route to db

volumes:
  pgdata:
SOLUTION
    echo ""
    echo "--- Verification ---"
    cat << 'SOLUTION'
# Start the stack
docker compose up -d

# Verify network assignments
docker inspect $(docker compose ps -q frontend) \
  --format '{{range $net, $config := .NetworkSettings.Networks}}{{$net}} {{end}}'
# Output: project_frontend-net
# frontend is ONLY on frontend-net

docker inspect $(docker compose ps -q backend) \
  --format '{{range $net, $config := .NetworkSettings.Networks}}{{$net}} {{end}}'
# Output: project_frontend-net project_backend-net
# backend bridges BOTH networks

docker inspect $(docker compose ps -q db) \
  --format '{{range $net, $config := .NetworkSettings.Networks}}{{$net}} {{end}}'
# Output: project_backend-net
# db is ONLY on backend-net

# Test: backend CAN reach db
docker compose exec backend ping -c 1 db
# PING db (172.x.x.x): 56 data bytes — SUCCESS

# Test: frontend CANNOT reach db
docker compose exec frontend ping -c 1 db 2>&1
# ping: bad address 'db' — EXPECTED FAILURE
# frontend has no route to the backend-net network

# Test: frontend CAN reach backend
docker compose exec frontend wget -qO- http://backend:3000
# Output: Hello from backend

# Cleanup
docker compose down -v
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
