# Project: Deployment Automation

**Difficulty**: ⭐⭐⭐⭐

**Previous**: [Project: Task Runner](./14_Project_Task_Runner.md) | **Next**: [Project: System Monitoring Tool](./16_Project_Monitor.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Build an SSH-based deployment script that uses rsync for file synchronization and remote command execution
2. Implement a symlink-based release directory structure that enables instant rollback to previous versions
3. Write a rolling deployment strategy that deploys to servers one at a time with health checks between each host
4. Configure health check probes that verify application responsiveness after each deployment step
5. Implement automatic rollback logic that reverts to the previous release when a health check fails
6. Write Docker entrypoint scripts with proper signal forwarding, configuration templating, and graceful shutdown
7. Manage environment-specific configuration using `.env` files, host inventories, and deployment strategies

---

Deploying code to production is where shell scripting skills have the highest leverage and the highest risk. A well-written deployment script can push changes to a fleet of servers in minutes with automatic rollback on failure; a poorly written one can take down your entire service. This project brings together SSH, rsync, signal handling, and error management to build deployment automation that you can adapt to real infrastructure -- from a single VPS to a multi-server production environment.

## 1. Overview

### What is Deployment Automation?

Deployment automation is the process of automatically moving code from source control to production servers. It eliminates manual steps, reduces errors, and enables faster, more reliable releases.

Key components include:

- **Remote execution**: Running commands on target servers via SSH
- **File synchronization**: Copying code and assets to servers
- **Health checks**: Verifying the deployment succeeded
- **Rollback capability**: Reverting to the previous version on failure
- **Multi-server orchestration**: Deploying to multiple hosts sequentially or in parallel

### Why Pure Bash?

While tools like Ansible, Terraform, and Kubernetes exist, bash deployment scripts offer:

1. **Zero dependencies**: No agents or orchestration tools required
2. **Transparency**: Exactly what runs on each server is visible
3. **Simplicity**: Perfect for small-to-medium deployments
4. **SSH-native**: Leverage existing SSH infrastructure
5. **Customization**: Easy to adapt for specific needs

### What We're Building

This lesson covers three deployment tools:

1. **SSH-based deployment**: Deploy to remote servers using rsync and SSH
2. **Rolling deployment**: Gradually deploy to a fleet of servers with health checks
3. **Docker entrypoint scripts**: Proper container initialization with signal handling

---

## 2. Design

### Architecture Overview

```
Deployment System
├── Configuration Management
│   ├── Environment variables (.env files)
│   ├── Target hosts (inventory)
│   └── Deployment strategy (rolling, blue-green)
│
├── Remote Execution
│   ├── SSH connection management
│   ├── Command execution on remote hosts
│   └── File synchronization (rsync)
│
├── Health Checks
│   ├── Application health endpoints
│   ├── Process verification
│   └── Log inspection
│
└── Rollback Strategy
    ├── Version tracking
    ├── Symlink-based releases
    └── Automatic rollback on failure
```

### Deployment Strategies

| Strategy | Description | Use Case | Risk |
|----------|-------------|----------|------|
| **All-at-once** | Deploy to all servers simultaneously | Low-traffic apps, staging | High |
| **Rolling** | Deploy to servers one-by-one or in batches | Production, gradual rollout | Medium |
| **Blue-Green** | Maintain two environments, switch traffic | Zero-downtime deploys | Low |
| **Canary** | Deploy to a subset, monitor, then full rollout | High-risk changes | Low |

We'll implement rolling deployment in this lesson.

### Target Directory Structure

On remote servers:

```
/opt/myapp/
├── current -> releases/20240215_143022/   # Symlink to active version
├── releases/
│   ├── 20240215_143022/                   # Current deployment
│   ├── 20240215_120000/                   # Previous deployment
│   └── 20240214_093000/                   # Older deployment
└── shared/
    ├── logs/
    ├── uploads/
    └── .env                                # Shared environment config
```

This structure enables:
- **Quick rollback**: Just change the symlink
- **Keep old versions**: Retain N previous releases
- **Shared state**: Logs and uploads persist across deploys

---

## 3. SSH-Based Deployment

### SSH Connection Basics

Establish passwordless SSH with key-based authentication:

```bash
# Generate SSH key (if not already done)
ssh-keygen -t ed25519 -f ~/.ssh/deploy_key -N ""

# Copy public key to remote server
ssh-copy-id -i ~/.ssh/deploy_key.pub user@server.example.com

# Test connection
ssh -i ~/.ssh/deploy_key user@server.example.com "echo 'Connected successfully'"
```

### Reusable SSH Functions

```bash
#!/usr/bin/env bash
set -euo pipefail

# SSH configuration
SSH_KEY="${SSH_KEY:-$HOME/.ssh/deploy_key}"
SSH_USER="${SSH_USER:-deploy}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Execute command on remote host
remote_exec() {
    local host="$1"
    shift
    local cmd="$*"

    ssh ${SSH_OPTS} -i "${SSH_KEY}" "${SSH_USER}@${host}" "${cmd}"
}

# Example usage
remote_exec "web01.example.com" "uptime"
remote_exec "web01.example.com" "df -h"
```

### File Synchronization with rsync

rsync is more efficient than `scp` for repeated deployments:

```bash
# Sync local directory to remote server
sync_files() {
    local host="$1"
    local source="$2"
    local destination="$3"

    rsync -avz --delete \
        -e "ssh ${SSH_OPTS} -i ${SSH_KEY}" \
        "${source}/" \
        "${SSH_USER}@${host}:${destination}/"
}

# Example: Deploy application code
sync_files "web01.example.com" "./build" "/opt/myapp/releases/$(date +%Y%m%d_%H%M%S)"
```

**rsync flags explained**:
- `-a`: Archive mode (preserves permissions, timestamps, etc.)
- `-v`: Verbose
- `-z`: Compress during transfer
- `--delete`: Remove files on destination that don't exist in source
- `-e`: Specify remote shell command

### Advanced: Excluding Files

```bash
sync_files_with_exclusions() {
    local host="$1"
    local source="$2"
    local destination="$3"

    rsync -avz --delete \
        --exclude='.git' \
        --exclude='node_modules' \
        --exclude='*.log' \
        --exclude='.env' \
        -e "ssh ${SSH_OPTS} -i ${SSH_KEY}" \
        "${source}/" \
        "${SSH_USER}@${host}:${destination}/"
}
```

### SSH Connection Pooling (ControlMaster)

Reuse SSH connections for faster repeated commands:

```bash
# Enable connection multiplexing
SSH_CONTROL_PATH="/tmp/ssh-control-%r@%h:%p"
SSH_OPTS+=" -o ControlMaster=auto -o ControlPath=${SSH_CONTROL_PATH} -o ControlPersist=10m"

# First command opens connection and keeps it alive for 10 minutes
remote_exec "web01.example.com" "echo 'First command'"

# Subsequent commands reuse the connection (much faster)
remote_exec "web01.example.com" "echo 'Second command'"
remote_exec "web01.example.com" "echo 'Third command'"

# Cleanup control socket when done
cleanup_ssh() {
    local host="$1"
    ssh ${SSH_OPTS} -O exit "${SSH_USER}@${host}" 2>/dev/null || true
}
```

---

## 4. Rolling Deployment

### Rolling Deployment Strategy

Deploy to servers sequentially, verifying each succeeds before proceeding:

1. Deploy to server 1
2. Run health check on server 1
3. If healthy, continue to server 2; otherwise rollback and abort
4. Repeat for all servers

### Implementation

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Rolling Deployment Script
# ============================================================================

# Configuration
APP_NAME="${APP_NAME:-myapp}"
APP_DIR="/opt/${APP_NAME}"
RELEASES_DIR="${APP_DIR}/releases"
CURRENT_LINK="${APP_DIR}/current"
SHARED_DIR="${APP_DIR}/shared"
KEEP_RELEASES=5

# Health check settings
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8080/health}"
HEALTH_CHECK_RETRIES=5
HEALTH_CHECK_DELAY=2

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${RESET} $*"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓${RESET} $*"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${RESET} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] !${RESET} $*"
}

# ============================================================================
# Remote Operations
# ============================================================================

remote_exec() {
    local host="$1"
    shift
    ssh ${SSH_OPTS} -i "${SSH_KEY}" "${SSH_USER}@${host}" "$@"
}

# ============================================================================
# Deployment Functions
# ============================================================================

# Create directory structure on remote server
setup_remote_directories() {
    local host="$1"

    log_info "Setting up directories on ${host}..."

    remote_exec "${host}" "mkdir -p ${RELEASES_DIR} ${SHARED_DIR}/logs"

    log_success "Directories ready on ${host}"
}

# Deploy application code to remote server
deploy_release() {
    local host="$1"
    local release_name="$(date +%Y%m%d_%H%M%S)"
    local release_path="${RELEASES_DIR}/${release_name}"

    log_info "Deploying release ${release_name} to ${host}..."

    # Create release directory
    remote_exec "${host}" "mkdir -p ${release_path}"

    # Sync application code
    rsync -avz --delete \
        --exclude='.git' \
        --exclude='*.log' \
        --exclude='.env' \
        -e "ssh ${SSH_OPTS} -i ${SSH_KEY}" \
        ./build/ \
        "${SSH_USER}@${host}:${release_path}/"

    # Link shared resources
    remote_exec "${host}" "ln -snf ${SHARED_DIR}/logs ${release_path}/logs"
    remote_exec "${host}" "ln -snf ${SHARED_DIR}/.env ${release_path}/.env"

    echo "${release_name}"
}

# Activate a release by updating the 'current' symlink
activate_release() {
    local host="$1"
    local release_name="$2"
    local release_path="${RELEASES_DIR}/${release_name}"

    log_info "Activating release ${release_name} on ${host}..."

    # Update symlink atomically
    remote_exec "${host}" "ln -snf ${release_path} ${CURRENT_LINK}"

    # Restart application
    restart_application "${host}"

    log_success "Activated ${release_name} on ${host}"
}

# Restart the application
restart_application() {
    local host="$1"

    log_info "Restarting application on ${host}..."

    # Systemd service restart
    if remote_exec "${host}" "systemctl is-active --quiet ${APP_NAME}"; then
        remote_exec "${host}" "sudo systemctl restart ${APP_NAME}"
    else
        log_warn "Service ${APP_NAME} not running on ${host}, starting..."
        remote_exec "${host}" "sudo systemctl start ${APP_NAME}"
    fi

    sleep 2
}

# ============================================================================
# Health Checks
# ============================================================================

check_health() {
    local host="$1"
    local retries="${HEALTH_CHECK_RETRIES}"

    log_info "Running health check on ${host}..."

    while [ "${retries}" -gt 0 ]; do
        if remote_exec "${host}" "curl -sf ${HEALTH_CHECK_URL} > /dev/null"; then
            log_success "Health check passed on ${host}"
            return 0
        fi

        retries=$((retries - 1))
        if [ "${retries}" -gt 0 ]; then
            log_warn "Health check failed, retrying in ${HEALTH_CHECK_DELAY}s... (${retries} attempts left)"
            sleep "${HEALTH_CHECK_DELAY}"
        fi
    done

    log_error "Health check failed on ${host} after ${HEALTH_CHECK_RETRIES} attempts"
    return 1
}

# ============================================================================
# Rollback
# ============================================================================

rollback() {
    local host="$1"

    log_warn "Rolling back deployment on ${host}..."

    # Get previous release
    local previous_release=$(remote_exec "${host}" \
        "ls -t ${RELEASES_DIR} | sed -n '2p'")

    if [ -z "${previous_release}" ]; then
        log_error "No previous release found for rollback on ${host}"
        return 1
    fi

    log_info "Rolling back to ${previous_release} on ${host}"

    # Reactivate previous release
    activate_release "${host}" "${previous_release}"

    if check_health "${host}"; then
        log_success "Rollback successful on ${host}"
        return 0
    else
        log_error "Rollback failed on ${host}"
        return 1
    fi
}

# ============================================================================
# Cleanup
# ============================================================================

cleanup_old_releases() {
    local host="$1"

    log_info "Cleaning up old releases on ${host}..."

    remote_exec "${host}" \
        "ls -t ${RELEASES_DIR} | tail -n +$((KEEP_RELEASES + 1)) | xargs -I {} rm -rf ${RELEASES_DIR}/{}"

    log_success "Cleanup complete on ${host}"
}

# ============================================================================
# Rolling Deploy to Fleet
# ============================================================================

rolling_deploy() {
    local hosts=("$@")

    log_info "Starting rolling deployment to ${#hosts[@]} servers..."
    local start_time=$(date +%s)
    local failed_hosts=()

    for host in "${hosts[@]}"; do
        log_info "Deploying to ${host}..."

        # Setup
        setup_remote_directories "${host}"

        # Deploy
        local release_name=$(deploy_release "${host}")

        # Activate
        activate_release "${host}" "${release_name}"

        # Health check
        if check_health "${host}"; then
            log_success "Deployment to ${host} successful"
            cleanup_old_releases "${host}"
        else
            log_error "Deployment to ${host} failed health check"

            # Rollback
            if rollback "${host}"; then
                log_warn "Rolled back ${host} successfully"
            fi

            failed_hosts+=("${host}")

            # Abort on first failure
            log_error "Aborting rolling deployment due to failure on ${host}"
            break
        fi

        log_info "Waiting 5 seconds before next deployment..."
        sleep 5
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    if [ ${#failed_hosts[@]} -eq 0 ]; then
        log_success "Rolling deployment completed successfully in ${duration}s"
        return 0
    else
        log_error "Rolling deployment failed on: ${failed_hosts[*]}"
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Load environment
    if [ -f .env.deploy ]; then
        source .env.deploy
    fi

    # Validate configuration
    if [ -z "${SSH_KEY:-}" ]; then
        log_error "SSH_KEY not set"
        exit 1
    fi

    if [ ! -f "${SSH_KEY}" ]; then
        log_error "SSH key not found: ${SSH_KEY}"
        exit 1
    fi

    # Read hosts from file or arguments
    local hosts=()

    if [ -f "hosts.txt" ]; then
        mapfile -t hosts < hosts.txt
    elif [ $# -gt 0 ]; then
        hosts=("$@")
    else
        log_error "No hosts specified. Provide hosts.txt or pass as arguments."
        exit 1
    fi

    # Run rolling deployment
    rolling_deploy "${hosts[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

---

## 5. Docker Entrypoint Scripts

### Why Proper Entrypoint Scripts Matter

Docker containers should:
- Handle signals gracefully (SIGTERM for shutdown)
- Wait for dependencies (database, Redis, etc.)
- Configure themselves from environment variables
- Execute as the correct user (not root)

A proper entrypoint script orchestrates initialization.

### Basic Entrypoint Structure

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Docker Entrypoint Script
# ============================================================================

# Signal handling for graceful shutdown
shutdown() {
    echo "Received SIGTERM, shutting down gracefully..."
    # Kill child processes
    kill -TERM "$APP_PID" 2>/dev/null || true
    wait "$APP_PID"
    exit 0
}

trap shutdown SIGTERM SIGINT

# ============================================================================
# Wait for Dependencies
# ============================================================================

wait_for_service() {
    local host="$1"
    local port="$2"
    local max_attempts=30
    local attempt=1

    echo "Waiting for ${host}:${port}..."

    while [ $attempt -le $max_attempts ]; do
        if nc -z "${host}" "${port}" 2>/dev/null; then
            echo "Service ${host}:${port} is ready!"
            return 0
        fi

        echo "Attempt ${attempt}/${max_attempts}: ${host}:${port} not ready, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "Service ${host}:${port} failed to become ready after ${max_attempts} attempts"
    return 1
}

# Wait for PostgreSQL
if [ -n "${POSTGRES_HOST:-}" ]; then
    wait_for_service "${POSTGRES_HOST}" "${POSTGRES_PORT:-5432}"
fi

# Wait for Redis
if [ -n "${REDIS_HOST:-}" ]; then
    wait_for_service "${REDIS_HOST}" "${REDIS_PORT:-6379}"
fi

# ============================================================================
# Configuration from Environment
# ============================================================================

# Substitute environment variables in config template
if [ -f /app/config.template.yml ]; then
    envsubst < /app/config.template.yml > /app/config.yml
    echo "Generated config.yml from environment variables"
fi

# ============================================================================
# Run Migrations (if needed)
# ============================================================================

if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
    echo "Running database migrations..."
    python manage.py migrate
fi

# ============================================================================
# Start Application
# ============================================================================

echo "Starting application..."

# Start app in background so we can handle signals
exec "$@" &
APP_PID=$!

# Wait for application process
wait "$APP_PID"
```

### Advanced: Template Processing with envsubst

Use `envsubst` to generate config files from environment variables:

```bash
# config.template.yml
database:
  host: ${DB_HOST}
  port: ${DB_PORT}
  user: ${DB_USER}
  password: ${DB_PASSWORD}

redis:
  host: ${REDIS_HOST}
  port: ${REDIS_PORT}

app:
  debug: ${DEBUG}
  secret_key: ${SECRET_KEY}
```

In the entrypoint:

```bash
export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5432}"
export REDIS_HOST="${REDIS_HOST:-localhost}"
export DEBUG="${DEBUG:-false}"

envsubst < config.template.yml > config.yml
```

### Wait-for-it Pattern

Reusable function for dependency waiting:

```bash
wait_for_it() {
    local service="$1"
    local timeout="${2:-30}"

    if [[ "${service}" =~ ^([^:]+):([0-9]+)$ ]]; then
        local host="${BASH_REMATCH[1]}"
        local port="${BASH_REMATCH[2]}"
    else
        echo "Invalid service format: ${service} (expected host:port)"
        return 1
    fi

    local start=$(date +%s)
    while true; do
        if nc -z "${host}" "${port}" 2>/dev/null; then
            echo "✓ ${service} is ready"
            return 0
        fi

        local now=$(date +%s)
        local elapsed=$((now - start))

        if [ $elapsed -ge $timeout ]; then
            echo "✗ ${service} not ready after ${timeout}s"
            return 1
        fi

        echo "Waiting for ${service}... (${elapsed}s)"
        sleep 1
    done
}

# Usage
wait_for_it "${DATABASE_HOST}:${DATABASE_PORT}" 60
wait_for_it "${REDIS_HOST}:${REDIS_PORT}" 30
```

### Running as Non-Root User

Security best practice: run application as non-root:

```bash
# In Dockerfile
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# Or in entrypoint (if you need root for initialization)
#!/bin/bash

# Do root-level setup
chown -R appuser:appuser /app/logs

# Drop privileges and execute app
exec gosu appuser "$@"
```

---

## 6. Environment Configuration

### Loading .env Files

```bash
load_env() {
    local env_file="${1:-.env}"

    if [ ! -f "${env_file}" ]; then
        echo "Warning: ${env_file} not found" >&2
        return 1
    fi

    # Export all variables from .env
    set -a
    source "${env_file}"
    set +a

    echo "Loaded environment from ${env_file}"
}

# Usage
load_env ".env.production"
```

### Environment Variable Validation

Ensure required variables are set:

```bash
validate_env() {
    local required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "SECRET_KEY"
        "API_KEY"
    )

    local missing_vars=()

    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("${var}")
        fi
    done

    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo "Error: Missing required environment variables:" >&2
        printf '  - %s\n' "${missing_vars[@]}" >&2
        return 1
    fi

    echo "✓ All required environment variables are set"
}

# Usage in entrypoint
load_env ".env"
validate_env || exit 1
```

### Secrets Management

Never hardcode secrets. Use environment variables or secret managers:

```bash
# Bad: Hardcoded secrets
DB_PASSWORD="super_secret_123"

# Good: From environment
DB_PASSWORD="${DB_PASSWORD}"

# Better: From secret file (Docker secrets, Kubernetes secrets)
if [ -f /run/secrets/db_password ]; then
    DB_PASSWORD="$(cat /run/secrets/db_password)"
fi

# Best: From secret manager (AWS Secrets Manager, Vault)
DB_PASSWORD="$(aws secretsmanager get-secret-value --secret-id prod/db/password --query SecretString --output text)"
```

---

## 7. Complete Deploy Script

Here's a full-featured deployment script combining all concepts:

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Complete Deployment Automation Script
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Configuration
APP_NAME="${APP_NAME:-myapp}"
DEPLOY_ENV="${DEPLOY_ENV:-production}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/deploy_key}"
SSH_USER="${SSH_USER:-deploy}"
BUILD_DIR="${PROJECT_ROOT}/build"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${RESET} $*"; }
success() { echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓${RESET} $*"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ✗${RESET} $*" >&2; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] !${RESET} $*"; }

# ============================================================================
# Preflight Checks
# ============================================================================

preflight_checks() {
    log "Running preflight checks..."

    # Check SSH key
    if [ ! -f "${SSH_KEY}" ]; then
        error "SSH key not found: ${SSH_KEY}"
        return 1
    fi

    # Check build directory
    if [ ! -d "${BUILD_DIR}" ]; then
        error "Build directory not found: ${BUILD_DIR}"
        return 1
    fi

    # Check hosts file
    if [ ! -f "${SCRIPT_DIR}/hosts.${DEPLOY_ENV}.txt" ]; then
        error "Hosts file not found: hosts.${DEPLOY_ENV}.txt"
        return 1
    fi

    # Verify build
    if [ ! -f "${BUILD_DIR}/index.html" ]; then
        warn "index.html not found in build directory"
    fi

    success "Preflight checks passed"
}

# ============================================================================
# Build
# ============================================================================

build_application() {
    log "Building application..."

    cd "${PROJECT_ROOT}"

    # Clean previous build
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"

    # Run build command (customize for your project)
    if [ -f "package.json" ]; then
        npm run build
    elif [ -f "Makefile" ]; then
        make build
    else
        error "No build system detected"
        return 1
    fi

    success "Build completed"
}

# ============================================================================
# Deploy to Single Host
# ============================================================================

deploy_to_host() {
    local host="$1"
    local release_name="$(date +%Y%m%d_%H%M%S)"

    log "Deploying ${release_name} to ${host}..."

    # Setup directories
    ssh -i "${SSH_KEY}" "${SSH_USER}@${host}" \
        "mkdir -p /opt/${APP_NAME}/{releases,shared/logs}"

    # Sync files
    rsync -avz --delete \
        --exclude='.git' \
        --exclude='node_modules' \
        --exclude='*.log' \
        -e "ssh -i ${SSH_KEY}" \
        "${BUILD_DIR}/" \
        "${SSH_USER}@${host}:/opt/${APP_NAME}/releases/${release_name}/"

    # Activate release
    ssh -i "${SSH_KEY}" "${SSH_USER}@${host}" \
        "ln -snf /opt/${APP_NAME}/releases/${release_name} /opt/${APP_NAME}/current"

    # Restart service
    ssh -i "${SSH_KEY}" "${SSH_USER}@${host}" \
        "sudo systemctl restart ${APP_NAME}"

    sleep 3

    # Health check
    if ssh -i "${SSH_KEY}" "${SSH_USER}@${host}" \
        "curl -sf http://localhost:8080/health >/dev/null"; then
        success "Deployment to ${host} successful"

        # Cleanup old releases
        ssh -i "${SSH_KEY}" "${SSH_USER}@${host}" \
            "ls -t /opt/${APP_NAME}/releases | tail -n +6 | xargs -I {} rm -rf /opt/${APP_NAME}/releases/{}"

        return 0
    else
        error "Health check failed on ${host}"
        return 1
    fi
}

# ============================================================================
# Rolling Deploy
# ============================================================================

rolling_deploy() {
    local hosts_file="${SCRIPT_DIR}/hosts.${DEPLOY_ENV}.txt"
    mapfile -t hosts < "${hosts_file}"

    log "Deploying to ${#hosts[@]} hosts in ${DEPLOY_ENV}..."

    for host in "${hosts[@]}"; do
        # Skip empty lines and comments
        [[ -z "${host}" || "${host}" =~ ^# ]] && continue

        if deploy_to_host "${host}"; then
            log "Waiting 10 seconds before next deployment..."
            sleep 10
        else
            error "Deployment failed on ${host}, aborting rollout"
            return 1
        fi
    done

    success "Rolling deployment completed successfully"
}

# ============================================================================
# Main
# ============================================================================

show_help() {
    cat <<EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
  build              Build the application
  deploy             Build and deploy to ${DEPLOY_ENV}
  rollback HOST      Rollback to previous release on HOST

Options:
  -e, --env ENV      Deployment environment (default: production)
  -h, --help         Show this help message

Environment Variables:
  APP_NAME           Application name (default: myapp)
  DEPLOY_ENV         Deployment environment (default: production)
  SSH_KEY            Path to SSH private key
  SSH_USER           SSH username (default: deploy)

Examples:
  $0 build
  $0 deploy
  $0 -e staging deploy
  $0 rollback web01.example.com
EOF
}

main() {
    local command=""

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            -e|--env)
                DEPLOY_ENV="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            build|deploy|rollback)
                command="$1"
                shift
                break
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Execute command
    case "${command}" in
        build)
            build_application
            ;;
        deploy)
            preflight_checks
            build_application
            rolling_deploy
            ;;
        rollback)
            if [ $# -eq 0 ]; then
                error "Rollback requires a host argument"
                exit 1
            fi
            # Rollback logic here
            ;;
        *)
            show_help
            exit 1
            ;;
    esac
}

main "$@"
```

---

## 8. Usage Examples

### Deploy to Production

```bash
# Build and deploy
./deploy.sh deploy

# Deploy to staging environment
./deploy.sh -e staging deploy
```

### Hosts File

Create `hosts.production.txt`:

```
web01.example.com
web02.example.com
web03.example.com
```

### Environment File

Create `.env.deploy`:

```bash
APP_NAME=myapp
DEPLOY_ENV=production
SSH_KEY=/home/user/.ssh/deploy_key
SSH_USER=deploy
HEALTH_CHECK_URL=http://localhost:8080/health
KEEP_RELEASES=5
```

### Docker Entrypoint

In `Dockerfile`:

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --production

COPY . .

COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
CMD ["node", "server.js"]
```

Run the container:

```bash
docker run \
  -e DB_HOST=postgres \
  -e DB_PORT=5432 \
  -e REDIS_HOST=redis \
  myapp:latest
```

---

## 9. Extensions

### 1. Blue-Green Deployment

Maintain two identical environments (blue and green):

```bash
blue_green_deploy() {
    local current_env=$(get_active_environment)
    local target_env=$( [ "${current_env}" = "blue" ] && echo "green" || echo "blue" )

    log "Current: ${current_env}, deploying to: ${target_env}"

    # Deploy to inactive environment
    deploy_to_environment "${target_env}"

    # Switch traffic
    switch_load_balancer_target "${target_env}"

    success "Traffic switched to ${target_env}"
}
```

### 2. Canary Deployment

Deploy to a subset of servers first:

```bash
canary_deploy() {
    local canary_hosts=("web01.example.com")
    local production_hosts=("web02.example.com" "web03.example.com")

    # Deploy to canary
    for host in "${canary_hosts[@]}"; do
        deploy_to_host "${host}"
    done

    # Monitor metrics
    log "Canary deployed. Monitor metrics for 10 minutes..."
    sleep 600

    # If metrics are good, deploy to production
    read -p "Proceed with full rollout? (y/n) " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rolling_deploy "${production_hosts[@]}"
    fi
}
```

### 3. Slack Notifications

Send deployment notifications:

```bash
send_slack_notification() {
    local message="$1"
    local webhook_url="${SLACK_WEBHOOK_URL}"

    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"${message}\"}" \
        "${webhook_url}"
}

# Usage
send_slack_notification "Deployment started to production"
rolling_deploy
send_slack_notification "Deployment to production completed successfully"
```

### 4. Deployment Locking

Prevent concurrent deployments:

```bash
acquire_lock() {
    local lock_file="/tmp/deploy.lock"

    if [ -f "${lock_file}" ]; then
        error "Another deployment is in progress"
        return 1
    fi

    echo $$ > "${lock_file}"
    trap "rm -f ${lock_file}" EXIT
}

# Usage
acquire_lock || exit 1
rolling_deploy
```

## Exercises

### Exercise 1: Implement a Health Check Function

Write a `health_check.sh` script that implements a robust health check function:

```bash
health_check() {
    local url="$1"
    local max_attempts="${2:-5}"
    local interval="${3:-3}"
    # Your implementation here
}
```

The function must:
- Use `curl -sf` to check the URL (silent mode, fail on HTTP errors)
- Retry up to `max_attempts` times with `interval` seconds between retries
- Print progress: `"Attempt 1/5: checking http://..."` on each retry
- Return 0 on success, 1 if all attempts fail
- Include a timeout of 5 seconds per attempt using `curl --max-time 5`

Test by pointing it at `http://httpbin.org/status/200` (should pass) and `http://httpbin.org/status/503` (should fail after retries).

### Exercise 2: Build a Symlink-Based Release Manager

Implement a local release management system that mimics the Capistrano-style directory structure:

```
releases/
    20240115_143022/   ← old release
    20240116_091500/   ← current release (symlinked)
current -> releases/20240116_091500
```

Write a `release.sh` script with three subcommands:
- `release.sh deploy <source_dir>` — copies `source_dir` to `releases/$(date +%Y%m%d_%H%M%S)/`, then updates the `current` symlink atomically using `ln -sfn`
- `release.sh rollback` — reads the two most recent releases, updates `current` to point to the second-newest, and prints which version was restored
- `release.sh list` — lists all releases with their timestamps and marks the current one with `*`

### Exercise 3: Add Environment-Specific Configuration

Extend the deployment script concept to support multiple environments. Create:
- A `configs/` directory with `staging.env` and `production.env` files, each setting `APP_PORT`, `DB_HOST`, `LOG_LEVEL`, and `HEALTH_URL`
- A `load_config <environment>` function that sources the appropriate `.env` file and validates that all required variables are set
- A deploy command that accepts `-e staging` or `-e production` and calls `load_config` before proceeding
- Prevent production deploys unless a `DEPLOY_CONFIRMED=yes` environment variable is set or the user types `yes` at an interactive confirmation prompt

### Exercise 4: Simulate a Rolling Deployment

Implement a rolling deployment simulation without real SSH by using local directories as "hosts":
- Create three directories `/tmp/server_{1,2,3}/app/` to simulate three servers
- Write a `rolling_deploy <version>` function that iterates over the three "servers" and for each one: creates a file `version.txt` with the version string, sleeps 1 second, then runs a fake health check
- If the fake health check returns failure (simulate by having server_2 fail), stop the deployment and roll back already-deployed servers by removing `version.txt`
- Print a summary at the end: which servers succeeded, which failed, and whether rollback occurred

### Exercise 5: Write a Docker Entrypoint Script

Create a production-quality `entrypoint.sh` for a web application container:
- Forward signals properly: trap `SIGTERM` and `SIGINT` to gracefully shut down the application process
- Template an `nginx.conf` from environment variables `SERVER_NAME`, `APP_PORT`, and `WORKER_PROCESSES` using `sed` or `envsubst`
- Validate required environment variables (`DATABASE_URL`, `SECRET_KEY`) at startup and exit with code 78 (configuration error) if any are missing
- After configuration validation, `exec` the main process (passed as `CMD` arguments via `"$@"`) so it becomes PID 1 and receives signals directly

---

**Previous**: [Project: Task Runner](./14_Project_Task_Runner.md) | **Next**: [Project: System Monitoring Tool](./16_Project_Monitor.md)
