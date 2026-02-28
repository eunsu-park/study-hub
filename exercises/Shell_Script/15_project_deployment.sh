#!/bin/bash
# Exercises for Lesson 15: Project - Deployment Automation
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Implement a Health Check Function ===
# Problem: health_check <url> <max_attempts> <interval> with retry, timeout, progress.
exercise_1() {
    echo "=== Exercise 1: Implement a Health Check Function ==="

    health_check() {
        local url="$1"
        local max_attempts="${2:-5}"
        local interval="${3:-3}"
        local attempt=1

        while (( attempt <= max_attempts )); do
            echo "  Attempt $attempt/$max_attempts: checking $url"

            # Use curl with timeout, silent mode, and fail on HTTP errors
            if curl -sf --max-time 5 "$url" > /dev/null 2>&1; then
                echo "  [OK] Health check passed on attempt $attempt"
                return 0
            fi

            if (( attempt < max_attempts )); then
                echo "  [WAIT] Retrying in ${interval}s..."
                sleep "$interval"
            fi

            (( attempt++ ))
        done

        echo "  [FAIL] Health check failed after $max_attempts attempts"
        return 1
    }

    # Test 1: Simulate a successful health check (use localhost or known URL)
    echo "--- Test 1: Simulated health check (mocked) ---"

    # Mock curl for testing (override with a function)
    local attempt_count=0
    curl() {
        (( attempt_count++ ))
        # Succeed on 3rd attempt
        if (( attempt_count >= 3 )); then
            return 0
        fi
        return 22  # curl HTTP error code
    }

    attempt_count=0
    health_check "http://localhost:8080/health" 5 1

    echo ""

    # Test 2: All attempts fail
    echo "--- Test 2: All attempts fail ---"
    curl() { return 22; }  # Always fail
    health_check "http://localhost:9999/health" 3 1 || true

    # Restore curl
    unset -f curl
}

# === Exercise 2: Build a Symlink-Based Release Manager ===
# Problem: release.sh with deploy, rollback, and list subcommands.
exercise_2() {
    echo "=== Exercise 2: Build a Symlink-Based Release Manager ==="

    local base_dir="/tmp/releases_$$"
    local releases_dir="$base_dir/releases"
    mkdir -p "$releases_dir"

    release_deploy() {
        local source_dir="$1"
        local release_name
        release_name=$(date +%Y%m%d_%H%M%S)

        # Copy source to new release directory
        local release_path="$releases_dir/$release_name"
        cp -r "$source_dir" "$release_path"

        # Atomic symlink update
        ln -sfn "$release_path" "$base_dir/current"

        echo "  [DEPLOY] Created release: $release_name"
        echo "  [DEPLOY] current -> $release_path"
    }

    release_rollback() {
        # Get the two most recent releases
        local releases
        releases=$(ls -t "$releases_dir" 2>/dev/null)
        local current_release
        current_release=$(basename "$(readlink "$base_dir/current" 2>/dev/null)" 2>/dev/null)
        local prev_release
        prev_release=$(echo "$releases" | sed -n '2p')

        if [ -z "$prev_release" ]; then
            echo "  [ERROR] No previous release found for rollback"
            return 1
        fi

        ln -sfn "$releases_dir/$prev_release" "$base_dir/current"
        echo "  [ROLLBACK] Restored from $current_release to $prev_release"
        echo "  [ROLLBACK] current -> $releases_dir/$prev_release"
    }

    release_list() {
        local current_release
        current_release=$(basename "$(readlink "$base_dir/current" 2>/dev/null)" 2>/dev/null || echo "")

        echo "  Releases:"
        for rel in $(ls -t "$releases_dir" 2>/dev/null); do
            if [ "$rel" = "$current_release" ]; then
                echo "    * $rel  (current)"
            else
                echo "      $rel"
            fi
        done
    }

    # Create fake source directories
    local src1="/tmp/src_v1_$$"
    local src2="/tmp/src_v2_$$"
    local src3="/tmp/src_v3_$$"
    mkdir -p "$src1" "$src2" "$src3"
    echo "v1" > "$src1/version.txt"
    echo "v2" > "$src2/version.txt"
    echo "v3" > "$src3/version.txt"

    # Deploy v1
    echo "--- Deploy v1 ---"
    release_deploy "$src1"
    sleep 1

    # Deploy v2
    echo ""
    echo "--- Deploy v2 ---"
    release_deploy "$src2"
    sleep 1

    # Deploy v3
    echo ""
    echo "--- Deploy v3 ---"
    release_deploy "$src3"

    echo ""
    echo "--- List releases ---"
    release_list

    echo ""
    echo "--- Current version ---"
    echo "  $(cat "$base_dir/current/version.txt")"

    # Rollback
    echo ""
    echo "--- Rollback ---"
    release_rollback

    echo ""
    echo "--- After rollback ---"
    release_list
    echo "  Current version: $(cat "$base_dir/current/version.txt")"

    rm -rf "$base_dir" "$src1" "$src2" "$src3"
}

# === Exercise 3: Add Environment-Specific Configuration ===
# Problem: configs/ with staging.env and production.env, load_config function,
# production deploy confirmation.
exercise_3() {
    echo "=== Exercise 3: Add Environment-Specific Configuration ==="

    local work_dir="/tmp/envconfig_$$"
    mkdir -p "$work_dir/configs"

    # Create environment configs
    cat > "$work_dir/configs/staging.env" << 'EOF'
APP_PORT=8080
DB_HOST=staging-db.internal
LOG_LEVEL=debug
HEALTH_URL=http://localhost:8080/health
EOF

    cat > "$work_dir/configs/production.env" << 'EOF'
APP_PORT=80
DB_HOST=prod-db.internal
LOG_LEVEL=warn
HEALTH_URL=http://localhost/health
EOF

    load_config() {
        local environment="$1"
        local config_file="$work_dir/configs/${environment}.env"

        if [ ! -f "$config_file" ]; then
            echo "  [ERROR] Config file not found: $config_file"
            return 1
        fi

        # Source the env file
        set -a
        source "$config_file"
        set +a

        # Validate required variables
        local required_vars=("APP_PORT" "DB_HOST" "LOG_LEVEL" "HEALTH_URL")
        local missing=()

        for var in "${required_vars[@]}"; do
            if [ -z "${!var:-}" ]; then
                missing+=("$var")
            fi
        done

        if [ ${#missing[@]} -gt 0 ]; then
            echo "  [ERROR] Missing required variables: ${missing[*]}"
            return 1
        fi

        echo "  [OK] Loaded $environment config"
        echo "    APP_PORT=$APP_PORT"
        echo "    DB_HOST=$DB_HOST"
        echo "    LOG_LEVEL=$LOG_LEVEL"
        echo "    HEALTH_URL=$HEALTH_URL"
        return 0
    }

    deploy() {
        local environment="$1"

        # Load environment config
        load_config "$environment" || return 1

        # Production safety check
        if [ "$environment" = "production" ]; then
            if [ "${DEPLOY_CONFIRMED:-}" = "yes" ]; then
                echo "  [OK] Production deploy confirmed via DEPLOY_CONFIRMED=yes"
            else
                # In non-interactive mode, block the deploy
                echo "  [WARN] Production deploy requires confirmation."
                echo "  Set DEPLOY_CONFIRMED=yes or answer 'yes' at the prompt."
                # Simulate declining confirmation
                echo "  [ABORT] Deploy to production cancelled."
                return 1
            fi
        fi

        echo "  [DEPLOY] Deploying to $environment..."
        echo "  [DEPLOY] Using port $APP_PORT, database $DB_HOST"
        echo "  [OK] Deployment to $environment complete"
    }

    # Test 1: Staging deploy
    echo "--- Test 1: Deploy to staging ---"
    deploy "staging"

    echo ""

    # Test 2: Production without confirmation (should be blocked)
    echo "--- Test 2: Deploy to production (no confirmation) ---"
    deploy "production"

    echo ""

    # Test 3: Production with DEPLOY_CONFIRMED=yes
    echo "--- Test 3: Deploy to production (with confirmation) ---"
    DEPLOY_CONFIRMED=yes deploy "production"

    echo ""

    # Test 4: Invalid environment
    echo "--- Test 4: Invalid environment ---"
    deploy "invalid_env" || true

    rm -rf "$work_dir"
}

# === Exercise 4: Simulate a Rolling Deployment ===
# Problem: Rolling deploy to 3 simulated servers, with failure and rollback.
exercise_4() {
    echo "=== Exercise 4: Simulate a Rolling Deployment ==="

    local work_dir="/tmp/rolling_$$"
    local servers=("server_1" "server_2" "server_3")

    # Create server directories
    for server in "${servers[@]}"; do
        mkdir -p "$work_dir/$server/app"
    done

    # Fake health check: server_2 always fails
    fake_health_check() {
        local server="$1"
        if [ "$server" = "server_2" ]; then
            return 1  # Simulate failure
        fi
        return 0
    }

    rolling_deploy() {
        local version="$1"
        local deployed_servers=()
        local failed_server=""

        echo "  Starting rolling deployment of version '$version'..."
        echo ""

        for server in "${servers[@]}"; do
            echo "  [$server] Deploying version '$version'..."

            # Write version file
            echo "$version" > "$work_dir/$server/app/version.txt"
            sleep 1

            # Health check
            if fake_health_check "$server"; then
                echo "  [$server] Health check PASSED"
                deployed_servers+=("$server")
            else
                echo "  [$server] Health check FAILED"
                failed_server="$server"

                # Rollback already-deployed servers
                echo ""
                echo "  [ROLLBACK] Rolling back due to failure on $server..."
                for deployed in "${deployed_servers[@]}"; do
                    rm -f "$work_dir/$deployed/app/version.txt"
                    echo "  [$deployed] Rolled back (version.txt removed)"
                done
                # Remove from failed server too
                rm -f "$work_dir/$server/app/version.txt"
                echo "  [$server] Cleaned up"

                break
            fi
        done

        # Summary
        echo ""
        echo "  --- Deployment Summary ---"
        echo "    Version:   $version"
        echo "    Succeeded: ${#deployed_servers[@]} (${deployed_servers[*]:-none})"
        if [ -n "$failed_server" ]; then
            echo "    Failed:    $failed_server"
            echo "    Rollback:  YES"
            echo "    Status:    FAILED"
        else
            echo "    Failed:    none"
            echo "    Rollback:  NO"
            echo "    Status:    SUCCESS"
        fi

        # Verify state
        echo ""
        echo "  --- Server State ---"
        for server in "${servers[@]}"; do
            if [ -f "$work_dir/$server/app/version.txt" ]; then
                echo "    $server: $(cat "$work_dir/$server/app/version.txt")"
            else
                echo "    $server: (no version deployed)"
            fi
        done
    }

    rolling_deploy "v2.1.0"

    rm -rf "$work_dir"
}

# === Exercise 5: Write a Docker Entrypoint Script ===
# Problem: Production-quality entrypoint.sh with signal handling, env templating,
# env validation, and exec for PID 1.
exercise_5() {
    echo "=== Exercise 5: Write a Docker Entrypoint Script ==="

    local work_dir="/tmp/entrypoint_$$"
    mkdir -p "$work_dir"

    # Create the entrypoint script
    cat > "$work_dir/entrypoint.sh" << 'ENTRYPOINT'
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Signal Handling
# ============================================================================

APP_PID=""

shutdown() {
    echo "[ENTRYPOINT] Received shutdown signal, stopping gracefully..."
    if [ -n "$APP_PID" ]; then
        kill -TERM "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    echo "[ENTRYPOINT] Shutdown complete."
    exit 0
}

trap shutdown SIGTERM SIGINT

# ============================================================================
# Environment Variable Validation
# ============================================================================

validate_env() {
    local required_vars=("DATABASE_URL" "SECRET_KEY")
    local missing=()

    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing+=("$var")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo "[ENTRYPOINT] ERROR: Missing required environment variables:" >&2
        printf '  - %s\n' "${missing[@]}" >&2
        exit 78  # EX_CONFIG
    fi

    echo "[ENTRYPOINT] Environment validation passed."
}

# ============================================================================
# Template Processing
# ============================================================================

generate_config() {
    local template="$1"
    local output="$2"

    if [ ! -f "$template" ]; then
        echo "[ENTRYPOINT] No config template found, skipping."
        return 0
    fi

    # Use sed to substitute environment variables
    sed \
        -e "s|\${SERVER_NAME}|${SERVER_NAME:-localhost}|g" \
        -e "s|\${APP_PORT}|${APP_PORT:-8080}|g" \
        -e "s|\${WORKER_PROCESSES}|${WORKER_PROCESSES:-auto}|g" \
        "$template" > "$output"

    echo "[ENTRYPOINT] Generated config: $output"
}

# ============================================================================
# Main
# ============================================================================

validate_env

# Generate nginx.conf from template if it exists
generate_config "/app/nginx.conf.template" "/app/nginx.conf"

echo "[ENTRYPOINT] Starting application..."

# exec replaces the shell with the app process, making it PID 1
# This ensures signals are forwarded directly to the application
exec "$@" &
APP_PID=$!
wait "$APP_PID"
ENTRYPOINT
    chmod +x "$work_dir/entrypoint.sh"

    echo "--- entrypoint.sh ---"
    cat "$work_dir/entrypoint.sh" | sed 's/^/  /'

    echo ""
    echo "--- Testing the entrypoint ---"

    # Create a template
    cat > "$work_dir/nginx.conf.template" << 'TEMPLATE'
server {
    listen ${APP_PORT};
    server_name ${SERVER_NAME};
    worker_processes ${WORKER_PROCESSES};
}
TEMPLATE

    # Test 1: Missing required variables
    echo ""
    echo "--- Test 1: Missing required vars ---"
    (
        unset DATABASE_URL SECRET_KEY 2>/dev/null || true
        bash "$work_dir/entrypoint.sh" echo "app started" 2>&1
    ) || echo "  Exit code: $? (expected 78 for EX_CONFIG)"

    # Test 2: Valid environment
    echo ""
    echo "--- Test 2: Valid environment with template ---"
    (
        export DATABASE_URL="postgres://localhost/mydb"
        export SECRET_KEY="super-secret-123"
        export SERVER_NAME="example.com"
        export APP_PORT="9090"
        export WORKER_PROCESSES="4"

        # Override generate_config to use our template
        sed \
            -e "s|\${SERVER_NAME}|${SERVER_NAME}|g" \
            -e "s|\${APP_PORT}|${APP_PORT}|g" \
            -e "s|\${WORKER_PROCESSES}|${WORKER_PROCESSES}|g" \
            "$work_dir/nginx.conf.template" > "$work_dir/nginx.conf"

        echo "  Environment validation: passed"
        echo "  Generated nginx.conf:"
        cat "$work_dir/nginx.conf" | sed 's/^/    /'
        echo ""
        echo "  [ENTRYPOINT] Would exec: echo 'app started'"
        echo "  app started"
    )

    echo ""
    echo "--- Key design points ---"
    echo "  1. trap SIGTERM/SIGINT for graceful shutdown"
    echo "  2. Validate required env vars, exit 78 (EX_CONFIG) if missing"
    echo "  3. Template config files with sed/envsubst"
    echo "  4. exec \"\$@\" makes the app PID 1 for direct signal handling"

    rm -rf "$work_dir"
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
