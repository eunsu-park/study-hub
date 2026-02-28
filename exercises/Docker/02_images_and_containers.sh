#!/bin/bash
# Exercises for Lesson 02: Images and Containers
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Image Exploration ===
# Problem: Pull python:3.11-slim and explore its structure.
exercise_1() {
    echo "=== Exercise 1: Image Exploration ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Pull the image
docker pull python:3.11-slim

# Step 2: List local images and note size
docker images python:3.11-slim
# REPOSITORY      TAG         IMAGE ID       CREATED        SIZE
# python          3.11-slim   abc123def456   2 weeks ago    ~155MB

# Step 3: View image history (layers)
docker history python:3.11-slim
# The output shows each layer (FROM, RUN, COPY, etc.) and its size.
# Typical layer count: ~8-10 layers
# Layers include: base Debian slim, Python runtime, pip, etc.

# Step 4: Inspect the image for exposed ports and default command
docker inspect python:3.11-slim --format '{{.Config.ExposedPorts}}'
# map[]  (no ports exposed by default)

docker inspect python:3.11-slim --format '{{.Config.Cmd}}'
# [python3]  (default command is the Python interpreter)

# Step 5: Compare with full python:3.11 image
docker pull python:3.11
docker images python
# REPOSITORY   TAG         SIZE
# python       3.11        ~920MB
# python       3.11-slim   ~155MB
#
# Why the difference:
# - python:3.11 is based on full Debian with build tools (gcc, make, etc.)
# - python:3.11-slim strips out build tools and docs
# - Slim is ~6x smaller — use it unless you need to compile C extensions
SOLUTION
}

# === Exercise 2: Container Lifecycle Management ===
# Problem: Practice the full container lifecycle using Nginx.
exercise_2() {
    echo "=== Exercise 2: Container Lifecycle Management ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Run Nginx in detached mode with a name and port mapping
docker run -d --name lifecycle-test -p 9090:80 nginx
# -d: run in background so the terminal is free
# --name: human-readable name for easier management
# -p 9090:80: host port 9090 forwards to container port 80

# Step 2: Verify the container is running
docker ps
# CONTAINER ID  IMAGE  COMMAND                 STATUS         PORTS                  NAMES
# abc123...     nginx  "/docker-entrypoint…"   Up 5 seconds   0.0.0.0:9090->80/tcp   lifecycle-test

# Step 3: Stop the container, confirm it is stopped
docker stop lifecycle-test
docker ps -a
# STATUS column now shows "Exited (0) X seconds ago"
# The container still exists but is no longer running

# Step 4: Start it again and verify
docker start lifecycle-test
docker ps
# STATUS: Up X seconds — the same container is reused, not a new one

# Step 5: View last 20 lines of logs
docker logs --tail 20 lifecycle-test
# Shows Nginx access/error log entries

# Step 6: Enter the container and check Nginx version
docker exec -it lifecycle-test nginx -v
# nginx version: nginx/1.25.x
# -it: allocate a TTY for interactive output

# Step 7: Force remove the container
docker rm -f lifecycle-test
# -f: force removes even a running container (sends SIGKILL)
# Without -f, you would need to stop it first
SOLUTION
}

# === Exercise 3: Volume Mount and Environment Variables ===
# Problem: Run PostgreSQL with persistent data and custom configuration.
exercise_3() {
    echo "=== Exercise 3: Volume Mount and Environment Variables ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Create a named volume
docker volume create pgdata
# Named volumes are managed by Docker and persist across container removal
# Data stored at /var/lib/docker/volumes/pgdata/_data on the host

# Step 2: Run PostgreSQL with custom env vars and the named volume
docker run -d \
  --name my-postgres \
  -e POSTGRES_USER=devuser \
  -e POSTGRES_PASSWORD=devpass \
  -e POSTGRES_DB=devdb \
  -v pgdata:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15-alpine
# -e: pass config at runtime — keeps the image generic and reusable
# -v pgdata:/var/lib/...  : named volume for data persistence

# Step 3: Verify the container is healthy
docker logs my-postgres
# Look for: "database system is ready to accept connections"
# PostgreSQL takes a few seconds to initialize on first run

# Step 4: Connect to PostgreSQL inside the container
docker exec -it my-postgres psql -U devuser -d devdb
# This opens an interactive psql session

# Step 5: List databases and exit
# Inside psql:
#   \l          -- lists all databases; devdb should appear
#   \q          -- quit psql

# Step 6: Verify data persistence across container removal
docker stop my-postgres
docker rm my-postgres
# Container is gone, but pgdata volume still exists

# Start a NEW container with the SAME volume
docker run -d \
  --name my-postgres-2 \
  -e POSTGRES_USER=devuser \
  -e POSTGRES_PASSWORD=devpass \
  -e POSTGRES_DB=devdb \
  -v pgdata:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15-alpine

# Connect and verify devdb still exists with data intact
docker exec -it my-postgres-2 psql -U devuser -d devdb -c '\l'
# devdb appears — data survived because the named volume was retained

# Cleanup
docker rm -f my-postgres-2
docker volume rm pgdata
SOLUTION
}

# === Exercise 4: Resource Monitoring and Cleanup ===
# Problem: Practice monitoring and cleaning up Docker resources.
exercise_4() {
    echo "=== Exercise 4: Resource Monitoring and Cleanup ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Start two containers
docker run -d --name web1 nginx
docker run -d --name web2 nginx

# Step 2: View resource usage (non-streaming snapshot)
docker stats --no-stream
# CONTAINER ID  NAME  CPU %  MEM USAGE / LIMIT  NET I/O   BLOCK I/O
# abc123...     web1  0.00%  3.5MiB / 16GiB     656B/0B   0B/0B
# def456...     web2  0.00%  3.5MiB / 16GiB     656B/0B   0B/0B
# Nginx is lightweight — uses very little CPU and ~3-4 MB of memory

# Step 3: Find container's IP address within the Docker network
docker inspect web1 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
# Example output: 172.17.0.2
# This is the IP on the default bridge network

# Step 4: Stop both containers
docker stop web1 web2

# Step 5: Confirm both are stopped
docker ps -a
# Both show STATUS: Exited (0) ...

# Step 6: Clean up all stopped containers
docker container prune
# This removes all stopped containers
# Type 'y' to confirm
# Output: Deleted Containers: abc123..., def456...

# Step 7: Remove the Nginx image
docker rmi nginx
# If stopped containers still reference the image, you'll get:
#   Error: image is referenced in multiple repositories
# Fix: ensure containers are removed first (prune did this in step 6)
# After prune, docker rmi nginx succeeds and frees disk space
SOLUTION
}

# === Exercise 5: Multi-Container Scenario ===
# Problem: Run a Node.js application with a Redis cache using a custom network.
exercise_5() {
    echo "=== Exercise 5: Multi-Container Scenario ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Create a Docker network for inter-container communication
docker network create app-net
# User-defined networks provide automatic DNS resolution by container name
# The default bridge network does NOT provide DNS — only IP-based access

# Step 2: Run Redis on the custom network
docker run -d --name redis-cache --network app-net redis:7-alpine
# redis:7-alpine is ~30 MB — much smaller than the full Redis image (~130 MB)

# Step 3: Run a Node.js container on the same network
docker run -it --rm \
  --name node-app \
  --network app-net \
  -v $(pwd):/app \
  -w /app \
  node:18-alpine \
  sh
# --rm: auto-remove container on exit (no leftover stopped containers)
# --network app-net: same network as redis — enables DNS-based discovery
# -v $(pwd):/app: mount current directory for code access
# -w /app: set working directory inside the container

# Step 4: Inside the Node.js container, verify DNS resolution
# In the container shell:
ping -c 2 redis-cache
# PING redis-cache (172.x.x.x): 56 data bytes
# 64 bytes from 172.x.x.x: seq=0 ttl=64 time=0.123 ms
# This works because both containers share the app-net network

# Step 5: Install Redis client and test connectivity
npm install redis
node -e "
const r = require('redis').createClient({url:'redis://redis-cache:6379'});
r.connect().then(() => {
  console.log('Connected to Redis!');
  r.quit();
})
"
# Output: Connected to Redis!
# 'redis-cache' resolves via Docker's embedded DNS server at 127.0.0.11

# Step 6: Clean up
exit  # Exit the Node.js container (--rm auto-removes it)
docker stop redis-cache
docker rm redis-cache
docker network rm app-net
# Always clean up custom networks to avoid resource leaks
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
