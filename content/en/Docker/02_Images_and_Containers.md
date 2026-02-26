# Docker Images and Containers

**Previous**: [Docker Basics](./01_Docker_Basics.md) | **Next**: [Dockerfile](./03_Dockerfile.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the layered structure of Docker images and how they are stored
2. Describe the image naming convention including registry, repository, and tag
3. Use Docker CLI commands to search, pull, list, inspect, and delete images
4. Run containers with options for port mapping, environment variables, volumes, and interactive mode
5. Manage container lifecycle operations: start, stop, restart, and remove
6. Access running containers, view logs, and monitor resource usage
7. Apply common option combinations for development and data persistence

---

Images and containers are the two most fundamental concepts in Docker. An image is a read-only blueprint that captures everything an application needs to run, while a container is a live, running instance of that image. Mastering how to manage images and containers through the Docker CLI is essential for daily development work, from pulling pre-built images off Docker Hub to running, inspecting, and cleaning up containers.

## 1. Docker Images

### What is an Image?

- **Read-only template** for creating containers
- Includes application + execution environment
- Efficiently stored in layer structure

### Image Name Structure

```
[registry/]repository:tag

Examples:
nginx                    → nginx:latest (default)
nginx:1.25              → specific version
node:18-alpine          → Node 18, Alpine Linux based
myname/myapp:v1.0       → user image
gcr.io/project/app:tag  → Google Container Registry
```

| Component | Description | Example |
|-----------|-------------|---------|
| Registry | Image repository | docker.io, gcr.io |
| Repository | Image name | nginx, node |
| Tag | Version | latest, 1.25, alpine |

---

## 2. Image Management Commands

### Search Images

```bash
# Search on Docker Hub
docker search nginx

# Output example:
# NAME          DESCRIPTION                 STARS   OFFICIAL
# nginx         Official build of Nginx     18000   [OK]
# bitnami/nginx Bitnami nginx Docker Image  150
```

### Download Images (Pull)

```bash
# Download latest version
docker pull nginx

# Download specific version — pin versions in production to avoid surprise breakages
docker pull nginx:1.25

# Alpine variant: ~175 MB vs ~1 GB full image — smaller attack surface, faster pulls
docker pull node:18-alpine
```

### List Images

```bash
# List local images
docker images

# Output example:
# REPOSITORY   TAG       IMAGE ID       CREATED        SIZE
# nginx        latest    a6bd71f48f68   2 days ago     187MB
# node         18-alpine 5d5f5d5f5d5f   1 week ago     175MB
```

### Delete Images

```bash
# Delete image
docker rmi nginx

# Delete by image ID
docker rmi a6bd71f48f68

# Force delete (image in use)
docker rmi -f nginx

# Delete all unused images — reclaims disk space from dangling (untagged) layers
docker image prune

# Delete all images (caution!)
docker rmi $(docker images -q)
```

### Image Details

```bash
# Image detailed information
docker inspect nginx

# Image history (check layers)
docker history nginx
```

---

## 3. Running Containers

### Basic Execution

```bash
# Basic run
docker run nginx

# -d: Detached mode — container runs in background, freeing the terminal
docker run -d nginx

# --name: Assign a human-readable name for easier management (logs, stop, exec)
docker run -d --name my-nginx nginx

# --rm: Auto-remove on exit — prevents accumulation of stopped containers
docker run --rm nginx
```

### Port Mapping (-p)

```bash
# -p host:container — forwards traffic from host port to the container's internal port
docker run -d -p 8080:80 nginx

# Multiple port mappings — e.g., HTTP and HTTPS on separate host ports
docker run -d -p 8080:80 -p 8443:443 nginx

# -P: Map all EXPOSEd ports to random high host ports (useful for quick testing)
docker run -d -P nginx
```

```
┌─────────────────────────────────────────────────────┐
│  Host (my computer)                                  │
│                                                     │
│  localhost:8080 ──────────────┐                     │
│                               │                     │
│  ┌────────────────────────────▼────────────────┐   │
│  │           Container (nginx)                  │   │
│  │                                             │   │
│  │           :80 (nginx default port)          │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Environment Variables (-e)

```bash
# -e passes config at runtime — keeps images generic and reusable across environments
docker run -d -e MYSQL_ROOT_PASSWORD=secret mysql

# Multiple environment variables
docker run -d \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=mydb \
  mysql
```

### Volume Mounting (-v)

```bash
# Bind mount — syncs host files into the container (useful for development)
docker run -d -v /host/path:/container/path nginx

# Mount current directory
docker run -d -v $(pwd):/app node

# :ro = read-only — container can read but not modify host files (security best practice)
docker run -d -v /host/path:/container/path:ro nginx

# Named volume — Docker manages the storage; data survives container removal
docker run -d -v mydata:/var/lib/mysql mysql
```

### Interactive Mode (-it)

```bash
# Access container shell
docker run -it ubuntu bash

# Inside container:
# root@container:/# ls
# root@container:/# exit
```

---

## 4. Container Management

### List Containers

```bash
# Running containers
docker ps

# All containers (including stopped)
docker ps -a

# Container IDs only
docker ps -q

# Output example:
# CONTAINER ID   IMAGE   COMMAND                  STATUS          PORTS                  NAMES
# abc123def456   nginx   "/docker-entrypoint.…"   Up 2 hours      0.0.0.0:8080->80/tcp   my-nginx
```

### Start/Stop/Restart Containers

```bash
# Stop
docker stop my-nginx

# Start (stopped container)
docker start my-nginx

# Restart
docker restart my-nginx

# Force kill
docker kill my-nginx
```

### Delete Containers

```bash
# Delete container (stopped only)
docker rm my-nginx

# Force delete (even if running)
docker rm -f my-nginx

# Delete all stopped containers
docker container prune

# Delete all containers (caution!)
docker rm -f $(docker ps -aq)
```

### Container Logs

```bash
# View logs
docker logs my-nginx

# Real-time logs (-f: follow)
docker logs -f my-nginx

# Last 100 lines
docker logs --tail 100 my-nginx

# Include timestamps
docker logs -t my-nginx
```

### Access Running Container

```bash
# Access container shell
docker exec -it my-nginx bash

# Execute specific command
docker exec my-nginx cat /etc/nginx/nginx.conf

# Access with root privileges
docker exec -it -u root my-nginx bash
```

### Container Information

```bash
# Detailed information
docker inspect my-nginx

# Resource usage
docker stats

# Real-time resource monitoring
docker stats my-nginx
```

---

## 5. Practice Examples

### Example 1: Nginx Web Server

```bash
# 1. Run Nginx container
docker run -d --name web -p 8080:80 nginx

# 2. Check in browser
# http://localhost:8080

# 3. Check logs
docker logs web

# 4. Access container
docker exec -it web bash

# 5. Check Nginx configuration
cat /etc/nginx/nginx.conf

# 6. Cleanup
exit
docker stop web
docker rm web
```

### Example 2: Serve Custom HTML

```bash
# 1. Create HTML file
mkdir -p ~/docker-test
echo "<h1>Hello Docker!</h1>" > ~/docker-test/index.html

# 2. Run with volume mount
docker run -d \
  --name my-web \
  -p 8080:80 \
  -v ~/docker-test:/usr/share/nginx/html:ro \
  nginx
# :ro — container serves files read-only; edits happen on the host only

# 3. Check in browser
# http://localhost:8080

# 4. Edit HTML (reflected in real-time)
echo "<h1>Updated!</h1>" > ~/docker-test/index.html

# 5. Cleanup
docker rm -f my-web
```

### Example 3: MySQL Database

```bash
# 1. Run MySQL container
docker run -d \
  --name mydb \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=testdb \
  -p 3306:3306 \
  mysql:8
# No named volume here — data is lost when the container is removed (fine for quick tests)

# 2. Check startup with logs — MySQL takes a few seconds to initialize; watch for "ready for connections"
docker logs -f mydb

# 3. Connect to MySQL client
docker exec -it mydb mysql -uroot -psecret

# 4. Inside MySQL:
# mysql> SHOW DATABASES;
# mysql> USE testdb;
# mysql> CREATE TABLE users (id INT, name VARCHAR(50));
# mysql> exit

# 5. Cleanup
docker rm -f mydb
```

### Example 4: Node.js Application

```bash
# 1. Create project directory
mkdir -p ~/node-docker
cd ~/node-docker

# 2. Create package.json
cat > package.json << 'EOF'
{
  "name": "docker-test",
  "version": "1.0.0",
  "main": "app.js",
  "scripts": {
    "start": "node app.js"
  }
}
EOF

# 3. Create app.js
cat > app.js << 'EOF'
const http = require('http');
const server = http.createServer((req, res) => {
  res.writeHead(200, {'Content-Type': 'text/plain'});
  res.end('Hello from Node.js in Docker!\n');
});
server.listen(3000, () => {
  console.log('Server running on port 3000');
});
EOF

# 4. Run container
docker run -d \
  --name node-app \
  -p 3000:3000 \
  -v $(pwd):/app \
  -w /app \
  node:18-alpine \
  node app.js
# -w /app: sets the working directory inside the container so 'node app.js' resolves correctly

# 5. Test
curl http://localhost:3000

# 6. Cleanup
docker rm -f node-app
```

---

## 6. Useful Option Combinations

### Development Environment

```bash
docker run -d \
  --name dev-server \
  -p 3000:3000 \
  -v $(pwd):/app \
  -w /app \
  --restart unless-stopped \
  node:18-alpine \
  npm run dev
# --restart unless-stopped: auto-restart on crash, but respect manual docker stop
# -v $(pwd):/app: bind mount enables live-reload — edit on host, see changes instantly
```

### Data Persistence

```bash
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15
# Named volume 'pgdata' — data survives container removal and can be backed up independently
```

---

## Command Summary

### Image Commands

| Command | Description |
|---------|-------------|
| `docker pull image` | Download image |
| `docker images` | List images |
| `docker rmi image` | Delete image |
| `docker image prune` | Delete unused images |

### Container Commands

| Command | Description |
|---------|-------------|
| `docker run` | Create and run container |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers |
| `docker stop` | Stop container |
| `docker start` | Start container |
| `docker rm` | Delete container |
| `docker logs` | View logs |
| `docker exec -it` | Access container |

### Key Options

| Option | Description |
|--------|-------------|
| `-d` | Run in background |
| `-p host:container` | Port mapping |
| `-v host:container` | Volume mount |
| `-e KEY=VALUE` | Environment variable |
| `--name` | Container name |
| `--rm` | Auto-remove on exit |
| `-it` | Interactive mode |

---

## Exercises

### Exercise 1: Image Exploration

Pull the `python:3.11-slim` image and explore its structure.

1. Pull the image: `docker pull python:3.11-slim`
2. List all local images and note the size of `python:3.11-slim`
3. Run `docker history python:3.11-slim` and count how many layers it has
4. Run `docker inspect python:3.11-slim` and find the exposed ports and default command
5. Compare the size of `python:3.11-slim` with `python:3.11` (full image). Which is larger and by how much?

### Exercise 2: Container Lifecycle Management

Practice the full container lifecycle using an Nginx container.

1. Run an Nginx container in detached mode named `lifecycle-test` on port 9090: `docker run -d --name lifecycle-test -p 9090:80 nginx`
2. Verify the container is running with `docker ps`
3. Stop the container, then confirm it is stopped with `docker ps -a`
4. Start the container again and verify it is running
5. View the last 20 lines of logs with `docker logs --tail 20 lifecycle-test`
6. Enter the container and check the Nginx version: `docker exec -it lifecycle-test nginx -v`
7. Remove the container forcefully with `docker rm -f lifecycle-test`

### Exercise 3: Volume Mount and Environment Variables

Run a PostgreSQL container with persistent data and custom configuration.

1. Create a named volume: `docker volume create pgdata`
2. Run the PostgreSQL container:
   ```bash
   docker run -d \
     --name my-postgres \
     -e POSTGRES_USER=devuser \
     -e POSTGRES_PASSWORD=devpass \
     -e POSTGRES_DB=devdb \
     -v pgdata:/var/lib/postgresql/data \
     -p 5432:5432 \
     postgres:15-alpine
   ```
3. Verify the container is healthy: `docker logs my-postgres`
4. Connect to PostgreSQL inside the container: `docker exec -it my-postgres psql -U devuser -d devdb`
5. Inside psql, run `\l` to list databases, then `\q` to exit
6. Stop and remove the container, then start a new one using the same `pgdata` volume and verify your `devdb` database persists

### Exercise 4: Resource Monitoring and Cleanup

Practice monitoring and cleaning up Docker resources.

1. Start two containers: `docker run -d --name web1 nginx` and `docker run -d --name web2 nginx`
2. Use `docker stats --no-stream` to view current resource usage for both containers
3. Use `docker inspect web1` to find its IP address within the Docker network
4. Stop both containers without removing them
5. Run `docker ps -a` to confirm both are in the stopped state
6. Clean up all stopped containers with `docker container prune` and confirm
7. Remove the Nginx image with `docker rmi nginx` and observe what happens if the image is referenced by a stopped container. Fix any errors.

### Exercise 5: Multi-Container Scenario

Run a Node.js application container that connects to a Redis cache.

1. Create a Docker network: `docker network create app-net`
2. Run Redis on the custom network:
   ```bash
   docker run -d --name redis-cache --network app-net redis:7-alpine
   ```
3. Run a Node.js container on the same network with source code mounted:
   ```bash
   docker run -it --rm \
     --name node-app \
     --network app-net \
     -v $(pwd):/app \
     -w /app \
     node:18-alpine \
     sh
   ```
4. Inside the container, verify DNS resolution: `ping -c 2 redis-cache`
5. Install the Redis client and test connectivity: `npm install redis && node -e "const r=require('redis').createClient({url:'redis://redis-cache:6379'});r.connect().then(()=>{console.log('Connected!');r.quit()})"`
6. Clean up: stop `redis-cache`, remove the `app-net` network

---

**Previous**: [Docker Basics](./01_Docker_Basics.md) | **Next**: [Dockerfile](./03_Dockerfile.md)
