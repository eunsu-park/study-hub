# Docker Practical Examples

**Previous**: [Docker Compose](./04_Docker_Compose.md) | **Next**: [Kubernetes Introduction](./06_Kubernetes_Intro.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Build a complete Node.js + Express + PostgreSQL application using Docker Compose
2. Implement multi-stage Docker builds for React applications served by Nginx
3. Compose full-stack applications with frontend, backend, database, and cache services
4. Configure WordPress with MySQL using Docker Compose for rapid CMS deployment
5. Apply debugging techniques including log monitoring, container access, and network inspection
6. Implement volume backup and restore strategies for persistent data

---

Knowing Docker commands and Compose syntax is only half the picture -- the real skill lies in applying them to actual projects. This lesson walks through four progressively complex real-world scenarios, from a simple API with a database to a full-stack application with React, Node.js, PostgreSQL, and Redis. By building these examples hands-on, you will develop the muscle memory and problem-solving instincts needed to Dockerize your own projects confidently.

---

## Example 1: Node.js + Express + PostgreSQL

### Project Structure

```
nodejs-postgres-app/
├── docker-compose.yml
├── .env
├── .dockerignore
├── backend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       └── index.js
└── db/
    └── init.sql
```

### Creating Files

**backend/package.json:**
```json
{
  "name": "express-postgres-app",
  "version": "1.0.0",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "node --watch src/index.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "pg": "^8.11.3"
  }
}
```

**backend/src/index.js:**
```javascript
const express = require('express');
const { Pool } = require('pg');

const app = express();
app.use(express.json());

// PostgreSQL connection
const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'myapp',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'password'
});

// Routes
app.get('/', (req, res) => {
  res.json({ message: 'Hello Docker!', status: 'running' });
});

app.get('/health', async (req, res) => {
  try {
    await pool.query('SELECT 1');
    res.json({ status: 'healthy', database: 'connected' });
  } catch (error) {
    res.status(500).json({ status: 'unhealthy', error: error.message });
  }
});

app.get('/users', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM users ORDER BY id');
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/users', async (req, res) => {
  const { name, email } = req.body;
  try {
    const result = await pool.query(
      'INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *',
      [name, email]
    );
    res.status(201).json(result.rows[0]);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

**backend/Dockerfile:**
```dockerfile
# Alpine: ~175 MB vs ~1 GB full image — smaller attack surface and faster CI pulls
FROM node:18-alpine

WORKDIR /app

# Copy dependency manifest first — changes less often, so Docker caches the install layer
COPY package*.json ./
# --production: skip devDependencies — smaller image and fewer potential vulnerabilities
RUN npm install --production

# Copy source code last — source changes don't invalidate the npm install cache
COPY . .

# Non-root user: limits damage if an attacker escapes the container
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

# Documentation only — does not actually publish the port (use -p for that)
EXPOSE 3000

# Exec form: process runs as PID 1, receives SIGTERM for graceful shutdown
CMD ["npm", "start"]
```

**backend/.dockerignore:**
```
node_modules
npm-debug.log
.git
.env
*.md
```

**db/init.sql:**
```sql
-- Create initial table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample data
INSERT INTO users (name, email) VALUES
    ('John Doe', 'john@example.com'),
    ('Jane Smith', 'jane@example.com'),
    ('Bob Johnson', 'bob@example.com');
```

**.env:**
```
DB_PASSWORD=secretpassword123
DB_USER=appuser
DB_NAME=myapp
```

**docker-compose.yml:**
```yaml
services:
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - DB_HOST=db             # Compose DNS resolves 'db' to the database container's IP
      - DB_PORT=5432
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}   # Read from .env file — keeps secrets out of version control
    depends_on:
      db:
        condition: service_healthy   # Wait until db passes health check, not just until it starts
    restart: unless-stopped          # Auto-restart on crash, but respect manual docker stop

  db:
    image: postgres:15-alpine        # Alpine variant: smaller image, faster pulls
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data          # Named volume — data survives container removal
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql  # Auto-runs on first start only
    healthcheck:
      # pg_isready verifies Postgres is accepting connections — not just that the process exists
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 5s
      timeout: 5s
      retries: 5
    ports:
      - "5432:5432"            # Expose to host for local DB tools (pgAdmin, DBeaver, etc.)

volumes:
  pgdata:
```

### Run and Test

```bash
# Create directories and navigate
mkdir -p nodejs-postgres-app/backend/src nodejs-postgres-app/db
cd nodejs-postgres-app

# (After creating above files)

# Run
docker compose up -d

# Check status
docker compose ps

# Check logs
docker compose logs -f backend

# API tests
curl http://localhost:3000/
curl http://localhost:3000/health
curl http://localhost:3000/users

# Add user
curl -X POST http://localhost:3000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Park", "email": "alice@example.com"}'

# Cleanup
docker compose down -v
```

---

## Example 2: React + Nginx (Production Build)

### Project Structure

```
react-nginx-app/
├── docker-compose.yml
├── Dockerfile
├── nginx.conf
├── package.json
├── public/
│   └── index.html
└── src/
    ├── App.js
    └── index.js
```

### Creating Files

**package.json:**
```json
{
  "name": "react-docker-app",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version"]
  }
}
```

**public/index.html:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>React Docker App</title>
</head>
<body>
  <div id="root"></div>
</body>
</html>
```

**src/index.js:**
```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
```

**src/App.js:**
```javascript
import React, { useState, useEffect } from 'react';

function App() {
  const [message, setMessage] = useState('Loading...');

  useEffect(() => {
    setMessage('Hello from React in Docker!');
  }, []);

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div style={{ textAlign: 'center' }}>
        <h1>{message}</h1>
        <p>This app is deployed with Docker.</p>
        <p>Build time: {new Date().toLocaleString()}</p>
      </div>
    </div>
  );
}

export default App;
```

**Dockerfile (Multi-stage build):**
```dockerfile
# Stage 1: Build — node_modules + build toolchain (~300 MB) are discarded after this stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy dependency manifest first — changes less often, so Docker caches the install layer
COPY package*.json ./
RUN npm install

# Copy source and build
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx — final image contains only static files (~25 MB)
FROM nginx:alpine

# --from=builder: pull artifacts from the build stage without carrying over node_modules
COPY --from=builder /app/build /usr/share/nginx/html

# Custom config for SPA routing, caching, and compression
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# "daemon off;" keeps nginx in the foreground so Docker can track the process as PID 1
CMD ["nginx", "-g", "daemon off;"]
```

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # React Router support (SPA) — all routes fall back to index.html so client-side routing works
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Static file caching — hashed filenames allow aggressive caching; "immutable" prevents revalidation
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # gzip compression — reduces transfer size by 60-80% for text-based assets
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
}
```

**docker-compose.yml:**
```yaml
services:
  frontend:
    build: .
    ports:
      - "80:80"
    restart: unless-stopped
```

### Run

```bash
# Build and run
docker compose up -d --build

# Check in browser
# http://localhost

# Cleanup
docker compose down
```

---

## Example 3: Full Stack (React + Node.js + PostgreSQL + Redis)

### Project Structure

```
fullstack-app/
├── docker-compose.yml
├── docker-compose.dev.yml
├── .env
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── (React project)
├── backend/
│   ├── Dockerfile
│   └── (Express project)
└── db/
    └── init.sql
```

**docker-compose.yml:**
```yaml
services:
  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped      # Auto-restart on crash, but respect manual docker stop

  # Backend API
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DB_HOST=db               # Compose DNS resolves service names to container IPs
      - DB_PORT=5432
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      db:
        condition: service_healthy    # Wait for Postgres to accept connections before starting
      redis:
        condition: service_started    # Redis starts fast — no health check needed
    restart: unless-stopped

  # PostgreSQL database
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data       # Named volume — data survives container removal
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      # pg_isready verifies Postgres is accepting connections — not just that the process exists
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis cache
  redis:
    image: redis:7-alpine
    # --appendonly yes: persist writes to disk — prevents data loss on restart (at slight perf cost)
    command: redis-server --appendonly yes
    volumes:
      - redisdata:/data
    restart: unless-stopped

volumes:
  pgdata:
  redisdata:
```

**docker-compose.dev.yml (Development override):**
```yaml
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev     # Dev Dockerfile may include hot-reload tooling
    ports:
      - "3001:3000"                  # Different host port avoids conflict with backend's :3000
    volumes:
      - ./frontend/src:/app/src      # Bind mount — edit on host, see changes instantly via hot-reload
    environment:
      - REACT_APP_API_URL=http://localhost:3000

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    volumes:
      - ./backend/src:/app/src       # Bind mount — enables live-reload for server code too
    environment:
      - NODE_ENV=development
    command: npm run dev             # Override CMD — use file-watching dev server instead of production start

  db:
    ports:
      - "5432:5432"                  # Expose to host so local DB tools can connect directly

  redis:
    ports:
      - "6379:6379"                  # Expose to host for redis-cli and debugging
```

### Run Commands

```bash
# Production
docker compose up -d

# Development
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Specific service logs
docker compose logs -f backend

# Database access
docker compose exec db psql -U ${DB_USER} -d ${DB_NAME}

# Redis CLI
docker compose exec redis redis-cli

# Full cleanup
docker compose down -v
```

---

## Example 4: WordPress + MySQL

### docker-compose.yml

```yaml
services:
  wordpress:
    image: wordpress:latest
    ports:
      - "8080:80"              # Non-standard host port to avoid conflicts if another service uses :80
    environment:
      - WORDPRESS_DB_HOST=db   # Compose DNS resolves 'db' to the MySQL container
      - WORDPRESS_DB_USER=wordpress
      - WORDPRESS_DB_PASSWORD=${DB_PASSWORD}
      - WORDPRESS_DB_NAME=wordpress
    volumes:
      - wordpress_data:/var/www/html   # Persist themes, plugins, and uploads across restarts
    depends_on:
      - db
    restart: unless-stopped    # Auto-restart on crash, but respect manual docker stop

  db:
    image: mysql:8
    environment:
      - MYSQL_DATABASE=wordpress
      - MYSQL_USER=wordpress
      - MYSQL_PASSWORD=${DB_PASSWORD}
      - MYSQL_ROOT_PASSWORD=${DB_ROOT_PASSWORD}   # Keep root password separate from app password
    volumes:
      - db_data:/var/lib/mysql         # Named volume — database files survive container removal
    restart: unless-stopped

  # phpMyAdmin (optional) — web-based DB admin for quick debugging; remove in production
  phpmyadmin:
    image: phpmyadmin:latest
    ports:
      - "8081:80"
    environment:
      - PMA_HOST=db
      - PMA_USER=wordpress
      - PMA_PASSWORD=${DB_PASSWORD}
    depends_on:
      - db

volumes:
  wordpress_data:
  db_data:
```

**.env:**
```
DB_PASSWORD=wordpresspass123
DB_ROOT_PASSWORD=rootpass123
```

### Run

```bash
docker compose up -d

# WordPress: http://localhost:8080
# phpMyAdmin: http://localhost:8081
```

---

## Useful Command Collection

### Debugging

```bash
# Access container
docker compose exec backend sh

# Real-time log monitoring
docker compose logs -f

# Check resource usage
docker stats

# Check network
docker network ls
docker network inspect <network_name>
```

### Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Full cleanup — removes ALL unused images, containers, networks, AND volumes (caution!)
docker system prune -a --volumes
```

### Backup

```bash
# Backup volume — uses a throwaway Alpine container to tar the volume contents
docker run --rm \
  -v pgdata:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/pgdata-backup.tar.gz -C /data .
# --rm: container auto-removes after the backup completes (no leftover containers)

# Restore volume — extracts the tarball into the named volume
docker run --rm \
  -v pgdata:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/pgdata-backup.tar.gz -C /data
```

---

## Exercises

### Exercise 1: Extend the Node.js + PostgreSQL Example

Build on Example 1 to add a new API endpoint and verify data persistence.

1. Follow Example 1 to get the Node.js + PostgreSQL stack running
2. Add a `DELETE /users/:id` endpoint to `backend/src/index.js` that deletes a user by ID
3. Rebuild only the backend image: `docker compose build backend`
4. Restart just the backend service: `docker compose up -d backend`
5. Use `curl -X DELETE http://localhost:3000/users/1` to delete a user
6. Verify the user is gone: `curl http://localhost:3000/users`
7. Run `docker compose down` without `-v` and restart — confirm the users table still has data

### Exercise 2: React + Nginx Multi-Stage Build Analysis

Analyze and optimize the React + Nginx multi-stage build from Example 2.

1. Follow Example 2 to build the React + Nginx image
2. Run `docker history <image-name>` to see all layers and their sizes
3. Run `docker images` and compare the final image size to a plain `node:18-alpine` image
4. Add a `.dockerignore` file to exclude `node_modules`, `.git`, and any test files; rebuild and compare sizes
5. Modify `nginx.conf` to add a `Cache-Control: no-store` header for `/index.html` and a 1-year cache for JS/CSS files
6. Rebuild and verify the headers with `curl -I http://localhost`

### Exercise 3: Full-Stack Debugging

Use the full-stack example (React + Node.js + PostgreSQL + Redis) to practice debugging.

1. Start the full-stack from Example 3
2. Identify which container is failing (if any) using `docker compose ps` and `docker compose logs`
3. Access the PostgreSQL database directly: `docker compose exec db psql -U $DB_USER -d $DB_NAME`
4. Access the Redis CLI: `docker compose exec redis redis-cli`
5. Inspect the network: `docker network inspect <project>_default` and document which containers are connected
6. Use `docker stats` to compare CPU and memory usage across all four services
7. Stop only the Redis service and observe how the backend handles cache unavailability

### Exercise 4: WordPress Volume Backup

Set up WordPress from Example 4 and practice data backup and restore.

1. Start the WordPress + MySQL stack from Example 4
2. Complete the WordPress installation through the browser at `http://localhost:8080`
3. Create a test blog post
4. Back up the `db_data` volume:
   ```bash
   docker run --rm \
     -v <project>_db_data:/data \
     -v $(pwd):/backup \
     alpine tar czf /backup/db-backup.tar.gz -C /data .
   ```
5. Run `docker compose down -v` to destroy all data
6. Restore the volume and restart the stack to verify the WordPress post is still there

### Exercise 5: Custom Full-Stack Project

Dockerize a project of your own choice using the patterns from this lesson.

1. Choose a simple application (e.g., a blog, a task manager, or a REST API) with at least two components (app + database)
2. Write a `Dockerfile` for each service following best practices: non-root user, layer caching, multi-stage if applicable
3. Write a `docker-compose.yml` with proper `depends_on`, health checks, named volumes, and a `.env` file for secrets
4. Add a `docker-compose.dev.yml` for local development with source code volume mounts
5. Document how to start the stack in the `README.md` with both dev and production commands
6. Verify data persists across `docker compose down` and `docker compose up` cycles

---

**Previous**: [Docker Compose](./04_Docker_Compose.md) | **Next**: [Kubernetes Introduction](./06_Kubernetes_Intro.md)
