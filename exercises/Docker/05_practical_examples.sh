#!/bin/bash
# Exercises for Lesson 05: Docker Practical Examples
# Topic: Docker
# Solutions to practice problems from the lesson.

# === Exercise 1: Extend the Node.js + PostgreSQL Example ===
# Problem: Add a DELETE endpoint and verify data persistence.
exercise_1() {
    echo "=== Exercise 1: Extend the Node.js + PostgreSQL Example ==="
    echo ""
    echo "Solution:"
    echo ""
    echo "--- Add DELETE endpoint to backend/src/index.js ---"
    cat << 'SOLUTION'
// Add this route to the existing Express app from Example 1:

app.delete('/users/:id', async (req, res) => {
  const { id } = req.params;
  try {
    const result = await pool.query(
      'DELETE FROM users WHERE id = $1 RETURNING *',
      [id]
    );
    if (result.rows.length === 0) {
      // No row matched the given ID — return 404 instead of 200
      // to distinguish "not found" from "successfully deleted"
      res.status(404).json({ error: 'User not found' });
    } else {
      res.json({ message: 'User deleted', user: result.rows[0] });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
SOLUTION
    echo ""
    echo "--- Commands ---"
    cat << 'SOLUTION'
# Step 3: Rebuild ONLY the backend image (db doesn't need rebuilding)
docker compose build backend
# This is faster than rebuilding everything with 'docker compose build'
# Only layers that changed (the COPY . . layer) are rebuilt

# Step 4: Restart just the backend service
docker compose up -d backend
# Compose replaces only the backend container; db keeps running
# The named volume keeps all database data intact during restart

# Step 5: Delete user with ID 1
curl -X DELETE http://localhost:3000/users/1
# {"message":"User deleted","user":{"id":1,"name":"John Doe",...}}

# Step 6: Verify the user is gone
curl http://localhost:3000/users
# Only Jane Smith and Bob Johnson remain

# Step 7: Test data persistence across restarts
docker compose down       # Stop and remove containers (NO -v flag!)
docker compose up -d      # Recreate containers

# Verify: users table still has data (minus the deleted user)
curl http://localhost:3000/users
# Jane Smith and Bob Johnson are still there
# Data persists because 'pgdata' named volume was NOT deleted

# Key insight: 'docker compose down' preserves volumes.
# Only 'docker compose down -v' destroys them.
# This is why named volumes are essential for databases.
SOLUTION
}

# === Exercise 2: React + Nginx Multi-Stage Build Analysis ===
# Problem: Analyze and optimize the React + Nginx multi-stage build.
exercise_2() {
    echo "=== Exercise 2: React + Nginx Multi-Stage Build Analysis ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 2: View image layers and sizes
docker history <image-name>
# IMAGE          CREATED        CREATED BY                          SIZE
# abc123...      2 min ago      COPY /app/build /usr/share/ngi...   1.5MB
# def456...      3 min ago      /bin/sh -c #(nop) COPY file:...     0B
# ...            ...            nginx:alpine base layers             ~25MB
#
# Notice: node_modules and build tools are NOT present in any layer.
# The multi-stage build discarded them after the build stage.

# Step 3: Compare with a plain node:18-alpine image
docker images
# REPOSITORY       TAG       SIZE
# react-nginx      latest    ~27MB    (multi-stage: nginx + static files)
# node             18-alpine ~175MB   (just the Node.js runtime!)
#
# The production image is ~6x smaller than even the base Node image.
# It contains ONLY nginx + the compiled static files.

# Step 4: Add .dockerignore to reduce build context
cat > .dockerignore << 'EOF'
node_modules
.git
*.test.js
*.spec.js
__tests__/
coverage/
.env
EOF
# Rebuild and compare — build context should be much smaller
# This speeds up the build by not sending node_modules to the daemon

# Step 5: Modify nginx.conf for proper caching headers
cat > nginx.conf << 'EOF'
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # SPA routing — client-side router handles all paths
    location / {
        try_files $uri $uri/ /index.html;
    }

    # index.html: NO CACHE
    # This is the entry point that references hashed JS/CSS files.
    # Must always be fresh so users get the latest asset references.
    location = /index.html {
        add_header Cache-Control "no-store, no-cache, must-revalidate";
    }

    # Static assets: AGGRESSIVE CACHE
    # Filenames contain content hashes (e.g., main.abc123.js)
    # so they are safe to cache forever — a new hash = a new URL
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff2?)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # gzip compression — reduces transfer size by 60-80%
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
}
EOF

# Step 6: Rebuild and verify headers
docker compose up -d --build
curl -I http://localhost
# index.html: Cache-Control: no-store, no-cache, must-revalidate
curl -I http://localhost/static/js/main.abc123.js
# JS files: Cache-Control: public, immutable, max-age=31536000
SOLUTION
}

# === Exercise 3: Full-Stack Debugging ===
# Problem: Debug a full-stack app with React, Node.js, PostgreSQL, and Redis.
exercise_3() {
    echo "=== Exercise 3: Full-Stack Debugging ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Start the full-stack application
docker compose up -d

# Step 2: Identify container status and any failures
docker compose ps
# NAME        SERVICE    STATUS           PORTS
# ...-frontend  frontend  running (healthy)  0.0.0.0:80->80/tcp
# ...-backend   backend   running           0.0.0.0:3000->3000/tcp
# ...-db        db        running (healthy)  5432/tcp
# ...-redis     redis     running           6379/tcp
#
# If a service shows "restarting" or "exited", check its logs:
docker compose logs backend
# Look for connection errors, missing env vars, or crash stack traces

# Step 3: Access PostgreSQL directly
docker compose exec db psql -U $DB_USER -d $DB_NAME
# Inside psql:
#   \dt          -- list tables
#   \l           -- list databases
#   SELECT * FROM users;
#   \q           -- quit
# This verifies the database schema and seed data are correct

# Step 4: Access Redis CLI
docker compose exec redis redis-cli
# Inside redis-cli:
#   PING         -- returns PONG
#   KEYS *       -- list all cached keys
#   GET mykey    -- retrieve a specific value
#   INFO memory  -- check memory usage
#   QUIT

# Step 5: Inspect the network
docker network ls
docker network inspect $(docker compose ps -q | head -1 | xargs docker inspect --format '{{range $net, $_ := .NetworkSettings.Networks}}{{$net}}{{end}}')
# Lists all containers on the default Compose network
# Each container gets a DNS entry matching its service name

# Step 6: Compare resource usage
docker stats --no-stream
# CONTAINER  CPU %  MEM USAGE / LIMIT   NET I/O    BLOCK I/O
# frontend   0.01%  5MiB / 16GiB        1kB/0B     0B/0B
# backend    0.05%  45MiB / 16GiB       2kB/1kB    0B/0B
# db         0.10%  28MiB / 16GiB       3kB/2kB    1MB/0B
# redis      0.02%  7MiB / 16GiB        1kB/0B     0B/0B
#
# PostgreSQL and the backend typically use the most memory
# Redis is very lightweight unless heavily loaded

# Step 7: Test Redis failure resilience
docker compose stop redis
# Backend should handle cache miss gracefully (fall back to DB)
curl http://localhost:3000/health
# If the backend crashes when Redis is down, you need:
#   - Try/catch around Redis operations
#   - Circuit breaker pattern
#   - Fallback to database or default values

# Restart Redis
docker compose start redis
SOLUTION
}

# === Exercise 4: WordPress Volume Backup ===
# Problem: Backup and restore WordPress database volumes.
exercise_4() {
    echo "=== Exercise 4: WordPress Volume Backup ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Step 1: Start WordPress + MySQL
docker compose up -d
# Wait for WordPress to be ready at http://localhost:8080

# Step 2: Complete WordPress installation through the browser
# Navigate to http://localhost:8080 and follow the setup wizard
# Choose language, set admin username/password, site title

# Step 3: Create a test blog post
# In the WordPress admin dashboard:
#   Posts > Add New > Title: "Test Post" > Content: "Hello!" > Publish

# Step 4: Backup the database volume
# This technique uses a throwaway Alpine container to tar the volume contents
docker run --rm \
  -v $(docker compose ps -q db | xargs docker inspect --format '{{range .Mounts}}{{if eq .Destination "/var/lib/mysql"}}{{.Name}}{{end}}{{end}}'):/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/db-backup.tar.gz -C /data .
# Breakdown:
# -v <volume>:/data        Mount the MySQL data volume at /data
# -v $(pwd):/backup        Mount current directory for output
# alpine tar czf ...       Create a compressed tarball of the volume
# --rm: container auto-removes after backup completes

# Alternatively, with a known volume name:
docker run --rm \
  -v myproject_db_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/db-backup.tar.gz -C /data .

# Step 5: Destroy everything
docker compose down -v
# All containers, networks, AND volumes are deleted
# WordPress data is gone — but we have the backup!

# Step 6: Restore from backup
# First, recreate the volume
docker volume create myproject_db_data

# Restore the tarball into the volume
docker run --rm \
  -v myproject_db_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/db-backup.tar.gz -C /data

# Restart the stack
docker compose up -d

# Navigate to http://localhost:8080
# The "Test Post" blog entry should still be there!
# WordPress installation, settings, and all content are restored.

# Key takeaway: Volume backups are essential for stateful services.
# Always test your restore procedure BEFORE you need it in production.
SOLUTION
}

# === Exercise 5: Custom Full-Stack Project ===
# Problem: Dockerize a project using patterns from the lesson.
exercise_5() {
    echo "=== Exercise 5: Custom Full-Stack Project ==="
    echo ""
    echo "Solution (Task Manager API example):"
    echo ""
    echo "--- Project Structure ---"
    cat << 'SOLUTION'
task-manager/
├── docker-compose.yml          # Production configuration
├── docker-compose.dev.yml      # Development overrides
├── .env                        # Secrets (never commit this!)
├── .dockerignore               # Exclude node_modules, .git, etc.
├── backend/
│   ├── Dockerfile              # Multi-stage or optimized single-stage
│   ├── package.json
│   └── src/
│       └── index.js
└── db/
    └── init.sql                # Schema and seed data
SOLUTION
    echo ""
    echo "--- backend/Dockerfile ---"
    cat << 'SOLUTION'
FROM node:18-alpine
WORKDIR /app

# Non-root user for security
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Layer caching: deps first, source last
COPY package*.json ./
RUN npm install --production

COPY . .
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 3000
CMD ["node", "src/index.js"]
SOLUTION
    echo ""
    echo "--- docker-compose.yml ---"
    cat << 'SOLUTION'
services:
  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - DB_HOST=db
      - DB_NAME=${DB_NAME}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  pgdata:
SOLUTION
    echo ""
    echo "--- docker-compose.dev.yml ---"
    cat << 'SOLUTION'
services:
  backend:
    volumes:
      - ./backend/src:/app/src    # Live-reload: edit on host, see changes instantly
    environment:
      - NODE_ENV=development
    command: node --watch src/index.js  # File-watching dev server
    ports:
      - "3000:3000"
      - "9229:9229"              # Node.js debugger port

  db:
    ports:
      - "5432:5432"              # Expose to host for local DB tools
SOLUTION
    echo ""
    echo "--- .env ---"
    cat << 'SOLUTION'
DB_NAME=taskmanager
DB_USER=taskuser
DB_PASSWORD=taskpass123
SOLUTION
    echo ""
    echo "--- Usage ---"
    cat << 'SOLUTION'
# Development
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
docker compose up -d

# Verify data persistence
docker compose down       # Containers removed, volume preserved
docker compose up -d      # Data still intact
docker compose exec db psql -U taskuser -d taskmanager -c "SELECT * FROM tasks;"
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
