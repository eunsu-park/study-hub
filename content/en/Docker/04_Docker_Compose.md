# Docker Compose

**Previous**: [Dockerfile](./03_Dockerfile.md) | **Next**: [Docker Practical Examples](./05_Practical_Examples.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what Docker Compose is and why it simplifies multi-container application management
2. Write a docker-compose.yml file with services, ports, environment variables, volumes, and networks
3. Use depends_on, healthcheck, and restart policies to manage service dependencies and reliability
4. Apply Docker Compose CLI commands to start, stop, scale, and monitor services
5. Configure environment-specific overrides using multiple Compose files
6. Implement service readiness patterns with health checks and conditional startup

---

Most real-world applications consist of multiple services -- a web server, a database, a cache, perhaps a message queue. Managing each of these as separate docker run commands quickly becomes unwieldy and error-prone. Docker Compose solves this by letting you define your entire application stack in a single YAML file and control it with one command. It is the standard tool for local development environments and simple production deployments.

## 1. What is Docker Compose?

Docker Compose is a tool for **defining and running multiple containers**. Manage entire application stacks with a single YAML file.

### Why use Docker Compose?

**Regular Docker commands:**
```bash
# Create network — needed so containers can reach each other by name
docker network create myapp-network

# Run database
docker run -d \
  --name db \
  --network myapp-network \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  postgres:15

# Run backend — must remember the exact network, env vars, volume for every service
docker run -d \
  --name backend \
  --network myapp-network \
  -e DATABASE_URL=postgres://... \
  -p 3000:3000 \
  my-backend

# Run frontend — three separate commands to manage; error-prone and hard to reproduce
docker run -d \
  --name frontend \
  --network myapp-network \
  -p 80:80 \
  my-frontend
```

**Docker Compose:**
```bash
docker compose up -d
```

| Advantage | Description |
|-----------|-------------|
| **Simplicity** | Run everything with one command |
| **Declarative** | Clearly defined in YAML |
| **Version control** | Manage config files with Git |
| **Reproducibility** | Reproduce identical environments |

---

## 2. Installation Check

Docker Compose is included with Docker Desktop.

```bash
# Check version
docker compose version
# Docker Compose version v2.23.0

# Or (old version)
docker-compose --version
```

> **Note:** `docker-compose` (with hyphen) is the old version, `docker compose` (with space) is the new version.

---

## 3. docker-compose.yml Basic Structure

```yaml
# docker-compose.yml

services:
  service-name1:
    image: image-name
    ports:
      - "host:container"
    environment:
      - variable=value
    volumes:
      - volume:path
    depends_on:
      - other-service

  service-name2:
    build: ./path
    ...

volumes:
  volume-name:

networks:
  network-name:
```

---

## 4. Main Configuration Options

### services - Define Services

```yaml
services:
  web:
    image: nginx:alpine
```

### image - Specify Image

```yaml
services:
  db:
    image: postgres:15

  redis:
    image: redis:7-alpine
```

### build - Build with Dockerfile

```yaml
services:
  app:
    build: .                    # Dockerfile in current directory

  api:
    build:
      context: ./backend        # Build context
      dockerfile: Dockerfile    # Dockerfile path
      args:                     # Build arguments
        - NODE_ENV=production
```

### ports - Port Mapping

```yaml
services:
  web:
    ports:
      - "8080:80"              # host:container
      - "443:443"

  api:
    ports:
      - "3000:3000"
```

### environment - Environment Variables

```yaml
services:
  db:
    environment:
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=secret
      - POSTGRES_DB=myapp

  # Or key: value format
  api:
    environment:
      NODE_ENV: production
      DB_HOST: db
```

### env_file - Environment Variable File

```yaml
services:
  api:
    env_file:
      - .env
      - .env.local
```

**.env file:**
```
DB_HOST=localhost
DB_PASSWORD=secret
API_KEY=abc123
```

### volumes - Volume Mounts

```yaml
services:
  db:
    volumes:
      - pgdata:/var/lib/postgresql/data    # Named volume — data survives container removal
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql  # Bind mount — auto-runs SQL on first start

  app:
    volumes:
      - ./src:/app/src                      # Source code mount — enables live-reload during dev
      - /app/node_modules                   # Anonymous volume — prevents host's node_modules from overwriting container's

volumes:
  pgdata:                                   # Declare here so Compose manages the volume lifecycle
```

### depends_on - Dependencies

```yaml
services:
  api:
    depends_on:
      - db
      - redis

  db:
    image: postgres:15

  redis:
    image: redis:7
```

> **Note:** `depends_on` only ensures startup order. It doesn't wait for the service to be "ready".

### networks - Networks

```yaml
services:
  frontend:
    networks:
      - frontend-net      # frontend can only talk to backend, not directly to db

  backend:
    networks:
      - frontend-net      # reachable by frontend
      - backend-net       # can reach db — acts as a gateway between the two networks

  db:
    networks:
      - backend-net       # isolated from frontend — reduces attack surface

networks:
  frontend-net:           # Separate networks enforce least-privilege network access
  backend-net:
```

### restart - Restart Policy

```yaml
services:
  web:
    restart: always              # Always restart — even after daemon reboot (production use)

  api:
    restart: unless-stopped      # Auto-restart on crash, but respect manual docker stop

  worker:
    restart: on-failure          # Restart only on non-zero exit — avoids infinite loops from intentional shutdowns
```

### healthcheck - Health Check

```yaml
services:
  api:
    healthcheck:
      # Orchestrators use health checks to restart unhealthy containers automatically
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s            # How often to probe
      timeout: 10s             # Max wait per probe before marking as failure
      retries: 3               # Consecutive failures before marking "unhealthy"
      start_period: 40s        # Grace period for slow-starting apps (failures don't count here)
```

---

## 5. Docker Compose Commands

### Run

```bash
# Run (foreground)
docker compose up

# Run in background
docker compose up -d

# Rebuild images then run
docker compose up --build

# Run specific services only
docker compose up -d web api
```

### Stop/Remove

```bash
# Stop
docker compose stop

# Stop and remove containers
docker compose down

# Also remove volumes — destroys persistent data; use only when you want a clean slate
docker compose down -v

# Also remove images — forces a fresh pull/build on next 'up'; useful after major changes
docker compose down --rmi all
```

### Check Status

```bash
# List services
docker compose ps

# View logs
docker compose logs

# View specific service logs
docker compose logs api

# Real-time logs
docker compose logs -f
```

### Service Management

```bash
# Restart
docker compose restart

# Restart specific service
docker compose restart api

# Scale services
docker compose up -d --scale api=3

# Execute command in service
docker compose exec api bash
docker compose exec db psql -U postgres
```

---

## 6. Practice Examples

### Example 1: Web + Database

**Project structure:**
```
my-webapp/
├── docker-compose.yml
├── .env
└── app/
    ├── Dockerfile
    └── index.js
```

**docker-compose.yml:**
```yaml
services:
  app:
    build: ./app
    ports:
      - "3000:3000"
    environment:
      # 'db' hostname works because Compose creates a shared network with DNS for each service
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      - db                       # Ensures db container starts first (but not necessarily "ready")

  db:
    image: postgres:15-alpine    # Alpine variant: smaller image, faster pulls
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    volumes:
      - pgdata:/var/lib/postgresql/data   # Named volume — data persists across restarts
    ports:
      - "5432:5432"              # Expose to host for local DB tools (pgAdmin, DBeaver, etc.)

volumes:
  pgdata:
```

**app/Dockerfile:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
# Copy dependency manifest first — changes less often, so Docker caches the install layer
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
# Exec form: process runs as PID 1, receives SIGTERM for graceful shutdown
CMD ["node", "index.js"]
```

**app/index.js:**
```javascript
const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.json({
    message: 'Hello from Docker Compose!',
    db_url: process.env.DATABASE_URL ? 'Connected' : 'Not set'
  });
});

app.listen(3000, () => console.log('Server on port 3000'));
```

**Run:**
```bash
cd my-webapp
docker compose up -d
curl http://localhost:3000
docker compose logs -f
docker compose down
```

### Example 2: Full Stack Application

```yaml
# docker-compose.yml

services:
  # Frontend (React)
  frontend:
    build: ./frontend
    ports:
      - "80:80"              # Standard HTTP port — no port prefix needed in browser URL
    depends_on:
      - backend

  # Backend (Node.js)
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DB_HOST=db           # Compose DNS resolves 'db' to the database container's IP
      - DB_NAME=myapp
      - REDIS_HOST=redis     # Same DNS-based discovery for the cache service
    depends_on:
      - db
      - redis

  # Database (PostgreSQL)
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=${DB_PASSWORD}   # Read from .env file — keeps secrets out of YAML
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql  # Auto-runs on first container start only

  # Cache (Redis)
  redis:
    image: redis:7-alpine                  # Alpine: ~30 MB vs ~130 MB full Redis image
    volumes:
      - redisdata:/data                    # Persist cache across restarts (useful for sessions)

  # Admin tool (pgAdmin)
  pgadmin:
    image: dpage/pgadmin4
    environment:
      - PGADMIN_DEFAULT_EMAIL=admin@example.com
      - PGADMIN_DEFAULT_PASSWORD=admin
    ports:
      - "5050:80"            # Non-standard host port to avoid conflicts with other services on :80
    depends_on:
      - db

volumes:
  pgdata:
  redisdata:
```

**.env:**
```
DB_PASSWORD=supersecret123
```

### Example 3: Development Environment

```yaml
# docker-compose.dev.yml

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev     # Separate Dockerfile — may include dev tools (nodemon, debugger)
    ports:
      - "3000:3000"
    volumes:
      - .:/app                    # Bind mount — edit on host, changes appear instantly in container
      - /app/node_modules         # Anonymous volume: prevents host bind mount from hiding container's installed modules
    environment:
      - NODE_ENV=development
    command: npm run dev          # Override CMD — use a file-watching dev server instead of production start

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=devpass
    ports:
      - "5432:5432"              # Expose to host so local DB tools (pgAdmin, psql) can connect directly
```

**Run:**
```bash
# Development environment
docker compose -f docker-compose.dev.yml up

# Production environment
docker compose -f docker-compose.yml up -d
```

---

## 7. Useful Patterns

### Environment-specific Configuration

```yaml
# docker-compose.yml (base)
services:
  app:
    image: myapp

# docker-compose.override.yml (dev, auto-merged)
services:
  app:
    build: .
    volumes:
      - .:/app

# docker-compose.prod.yml (production)
services:
  app:
    restart: always
```

```bash
# Development: auto-merges docker-compose.yml + docker-compose.override.yml
docker compose up

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Service Wait (wait-for-it)

```yaml
services:
  app:
    depends_on:
      db:
        condition: service_healthy   # Wait until db is actually ready, not just started

  db:
    image: postgres:15
    healthcheck:
      # pg_isready checks if Postgres is accepting connections — better than just checking if the process is alive
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
```

---

## Command Summary

| Command | Description |
|---------|-------------|
| `docker compose up` | Start services |
| `docker compose up -d` | Start in background |
| `docker compose up --build` | Rebuild then start |
| `docker compose down` | Stop and remove services |
| `docker compose down -v` | Also remove volumes |
| `docker compose ps` | Service status |
| `docker compose logs` | View logs |
| `docker compose logs -f` | Real-time logs |
| `docker compose exec service command` | Execute command |
| `docker compose restart` | Restart |

---

## Exercises

### Exercise 1: Two-Service Stack

Create a Docker Compose stack with a simple web app and a Redis counter.

1. Write a `docker-compose.yml` with two services:
   - `redis`: using the `redis:7-alpine` image
   - `web`: using `python:3.11-slim`, with port 5000 published, and `DATABASE_URL=redis://redis:6379` as an environment variable
2. Add a `depends_on` rule so `web` starts after `redis`
3. Run `docker compose up -d` and verify both services are running with `docker compose ps`
4. View logs for the `redis` service: `docker compose logs redis`
5. Execute a Redis CLI command inside the `redis` container: `docker compose exec redis redis-cli ping`
6. Tear down with `docker compose down` and confirm all containers are removed

### Exercise 2: Persistent Database with Health Check

Configure a PostgreSQL service with a health check and dependent app startup.

1. Create a `docker-compose.yml` with:
   - `db`: `postgres:15-alpine` with a named volume `pgdata:/var/lib/postgresql/data` and a `healthcheck` using `pg_isready`
   - `app`: any image, with `depends_on.db.condition: service_healthy`
2. Start the stack and run `docker compose ps` — observe that `app` only starts after `db` is healthy
3. Stop the stack and restart it; confirm `db` does not lose data because of the named volume
4. Run `docker compose down -v` and observe that the named volume is also deleted

### Exercise 3: Development vs Production Environments

Use multiple Compose files to manage environment-specific configuration.

1. Create a base `docker-compose.yml` with a `web` service using a built image (`build: .`)
2. Create `docker-compose.override.yml` for development:
   - Mount source code as a volume: `.:/app`
   - Set `NODE_ENV=development`
   - Map port `3001:3000`
3. Create `docker-compose.prod.yml` for production:
   - Add `restart: always`
   - Set `NODE_ENV=production`
   - Map port `80:3000`
4. Start in dev mode: `docker compose up` (auto-merges override)
5. Start in prod mode: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`
6. Verify the differences in configuration between the two modes

### Exercise 4: Service Scaling

Scale a service and observe load distribution.

1. Create a `docker-compose.yml` with an `api` service that responds with its hostname (use `hashicorp/http-echo -text="$(hostname)"` or a similar image)
2. Start with `docker compose up -d`
3. Scale the `api` service to 3 replicas: `docker compose up -d --scale api=3`
4. Verify three containers are running with `docker compose ps`
5. Use `docker compose logs api` to see logs from all replicas
6. Scale back down to 1 replica and confirm

### Exercise 5: Full-Stack Application Compose

Create a compose file for a three-service stack: frontend, backend, and database.

1. Define three services in `docker-compose.yml`:
   - `db`: `postgres:15-alpine` with environment variables and a named volume
   - `backend`: built from a local Dockerfile, depends on `db`, with database connection env vars
   - `frontend`: built from another Dockerfile, depends on `backend`, with port 80 published
2. Define two networks: `frontend-net` (frontend + backend) and `backend-net` (backend + db)
3. Assign each service to the appropriate network(s) so `frontend` cannot directly reach `db`
4. Start the stack, use `docker compose exec db psql` to verify the database is accessible from `backend` but not from `frontend`
5. Use `docker inspect` to confirm the network assignments

---

**Previous**: [Dockerfile](./03_Dockerfile.md) | **Next**: [Docker Practical Examples](./05_Practical_Examples.md)
