# 프로젝트: 배포 자동화(Deployment Automation)

**난이도**: ⭐⭐⭐⭐

**이전**: [14_Project_Task_Runner.md](./14_Project_Task_Runner.md) | **다음**: [16_Project_Monitor.md](./16_Project_Monitor.md)

---

## 1. 개요

### 배포 자동화란?

배포 자동화는 소스 제어에서 프로덕션 서버로 코드를 자동으로 이동하는 프로세스입니다. 수동 단계를 제거하고 에러를 줄이며 더 빠르고 신뢰할 수 있는 릴리스를 가능하게 합니다.

주요 구성 요소는 다음과 같습니다:

- **원격 실행**: SSH를 통해 대상 서버에서 명령어 실행
- **파일 동기화**: 서버로 코드와 자산 복사
- **헬스 체크**: 배포가 성공했는지 확인
- **롤백 기능**: 실패 시 이전 버전으로 되돌리기
- **다중 서버 오케스트레이션**: 여러 호스트에 순차적 또는 병렬로 배포

### 왜 순수 Bash인가?

Ansible, Terraform, Kubernetes와 같은 도구가 존재하지만, bash 배포 스크립트는 다음을 제공합니다:

1. **제로 의존성**: 에이전트나 오케스트레이션 도구 불필요
2. **투명성**: 각 서버에서 실행되는 것이 정확히 보임
3. **단순성**: 소규모에서 중규모 배포에 완벽
4. **SSH 네이티브**: 기존 SSH 인프라 활용
5. **커스터마이징**: 특정 요구사항에 맞게 쉽게 적응

### 우리가 만들 것

이 레슨은 세 가지 배포 도구를 다룹니다:

1. **SSH 기반 배포**: rsync와 SSH를 사용하여 원격 서버에 배포
2. **롤링 배포**: 헬스 체크와 함께 서버 플릿에 점진적으로 배포
3. **Docker 엔트리포인트 스크립트**: 시그널 처리를 통한 적절한 컨테이너 초기화

---

## 2. 설계

### 아키텍처 개요

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

### 배포 전략

| 전략 | 설명 | 사용 사례 | 위험도 |
|----------|-------------|----------|------|
| **All-at-once** | 모든 서버에 동시에 배포 | 낮은 트래픽 앱, 스테이징 | 높음 |
| **Rolling** | 서버에 하나씩 또는 배치로 배포 | 프로덕션, 점진적 롤아웃 | 중간 |
| **Blue-Green** | 두 환경 유지, 트래픽 전환 | 제로 다운타임 배포 | 낮음 |
| **Canary** | 하위 집합에 배포, 모니터링, 전체 롤아웃 | 높은 위험 변경 | 낮음 |

이 레슨에서는 롤링 배포를 구현합니다.

### 대상 디렉터리 구조

원격 서버에서:

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

이 구조는 다음을 가능하게 합니다:
- **빠른 롤백**: 심볼릭 링크만 변경
- **이전 버전 유지**: N개의 이전 릴리스 보유
- **공유 상태**: 로그와 업로드가 배포 간 유지됨

---

## 3. SSH 기반 배포

### SSH 연결 기본

키 기반 인증으로 암호 없는 SSH 설정:

```bash
# Generate SSH key (if not already done)
ssh-keygen -t ed25519 -f ~/.ssh/deploy_key -N ""

# Copy public key to remote server
ssh-copy-id -i ~/.ssh/deploy_key.pub user@server.example.com

# Test connection
ssh -i ~/.ssh/deploy_key user@server.example.com "echo 'Connected successfully'"
```

### 재사용 가능한 SSH 함수

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

### rsync를 통한 파일 동기화

rsync는 반복적인 배포에서 `scp`보다 효율적입니다:

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

**rsync 플래그 설명**:
- `-a`: 아카이브 모드 (권한, 타임스탬프 등 보존)
- `-v`: 상세 출력
- `-z`: 전송 중 압축
- `--delete`: 소스에 없는 파일을 대상에서 제거
- `-e`: 원격 셸 명령어 지정

### 고급: 파일 제외

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

### SSH 연결 풀링 (ControlMaster)

반복되는 명령어를 위해 SSH 연결 재사용:

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

## 4. 롤링 배포

### 롤링 배포 전략

서버에 순차적으로 배포하고, 계속 진행하기 전에 각각이 성공했는지 확인:

1. 서버 1에 배포
2. 서버 1에서 헬스 체크 실행
3. 정상이면 서버 2로 계속; 그렇지 않으면 롤백하고 중단
4. 모든 서버에 대해 반복

### 구현

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

## 5. Docker 엔트리포인트 스크립트

### 적절한 엔트리포인트 스크립트가 중요한 이유

Docker 컨테이너는 다음을 수행해야 합니다:
- 시그널을 우아하게 처리 (종료를 위한 SIGTERM)
- 의존성 대기 (데이터베이스, Redis 등)
- 환경 변수로부터 자신을 구성
- 올바른 사용자로 실행 (root가 아님)

적절한 엔트리포인트 스크립트는 초기화를 오케스트레이션합니다.

### 기본 엔트리포인트 구조

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

### 고급: envsubst를 사용한 템플릿 처리

환경 변수로부터 설정 파일을 생성하기 위해 `envsubst` 사용:

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

엔트리포인트에서:

```bash
export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5432}"
export REDIS_HOST="${REDIS_HOST:-localhost}"
export DEBUG="${DEBUG:-false}"

envsubst < config.template.yml > config.yml
```

### Wait-for-it 패턴

의존성 대기를 위한 재사용 가능한 함수:

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

### 비루트 사용자로 실행

보안 모범 사례: 애플리케이션을 비루트로 실행:

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

## 6. 환경 설정

### .env 파일 로드

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

### 환경 변수 검증

필수 변수가 설정되었는지 확인:

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

### 비밀 관리

비밀을 하드코딩하지 마세요. 환경 변수나 비밀 관리자를 사용하세요:

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

## 7. 완전한 배포 스크립트

모든 개념을 결합한 완전한 기능의 배포 스크립트입니다:

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

## 8. 사용 예시

### 프로덕션에 배포

```bash
# Build and deploy
./deploy.sh deploy

# Deploy to staging environment
./deploy.sh -e staging deploy
```

### 호스트 파일

`hosts.production.txt` 생성:

```
web01.example.com
web02.example.com
web03.example.com
```

### 환경 파일

`.env.deploy` 생성:

```bash
APP_NAME=myapp
DEPLOY_ENV=production
SSH_KEY=/home/user/.ssh/deploy_key
SSH_USER=deploy
HEALTH_CHECK_URL=http://localhost:8080/health
KEEP_RELEASES=5
```

### Docker 엔트리포인트

`Dockerfile`에서:

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

컨테이너 실행:

```bash
docker run \
  -e DB_HOST=postgres \
  -e DB_PORT=5432 \
  -e REDIS_HOST=redis \
  myapp:latest
```

---

## 9. 확장

### 1. Blue-Green 배포

두 개의 동일한 환경(blue와 green) 유지:

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

### 2. Canary 배포

먼저 서버의 하위 집합에 배포:

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

### 3. Slack 알림

배포 알림 전송:

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

### 4. 배포 잠금

동시 배포 방지:

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

---

**이전**: [14_Project_Task_Runner.md](./14_Project_Task_Runner.md) | **다음**: [16_Project_Monitor.md](./16_Project_Monitor.md)
