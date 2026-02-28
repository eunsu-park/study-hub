#!/bin/bash
# Exercises for Lesson 14: Project - Task Runner
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Add a New Task to the Runner ===
# Problem: Add task::docs and task::run tasks that depend on build.
exercise_1() {
    echo "=== Exercise 1: Add a New Task to the Runner ==="

    # Mini task runner framework (simplified for exercise)
    declare -A TASK_DEPS
    declare -A TASK_EXECUTED

    log_info()    { echo "  [$(date +%H:%M:%S)] $*"; }
    log_success() { echo "  [$(date +%H:%M:%S)] [OK] $*"; }

    depends_on() {
        for dep in "$@"; do
            if [ -z "${TASK_EXECUTED[$dep]:-}" ]; then
                execute_task "$dep"
            fi
        done
    }

    execute_task() {
        local task="$1"
        if [ -n "${TASK_EXECUTED[$task]:-}" ]; then
            return 0
        fi
        log_info "Running task: $task"
        "task_$task"
        TASK_EXECUTED["$task"]=1
        log_success "Task '$task' completed"
    }

    # Existing tasks
    task_clean() {
        log_info "Cleaning build directory..."
        rm -rf /tmp/taskrunner_build_$$ 2>/dev/null || true
    }

    task_deps() {
        depends_on "clean"
        log_info "Installing dependencies..."
    }

    task_build() {
        depends_on "deps"
        log_info "Compiling source files..."
        mkdir -p /tmp/taskrunner_build_$$
        echo "built" > /tmp/taskrunner_build_$$/app
    }

    # --- NEW: task::docs ---
    task_docs() {
        depends_on "build"
        log_info "Generating docs..."
        mkdir -p /tmp/taskrunner_build_$$/docs
        echo "API Reference" > /tmp/taskrunner_build_$$/docs/index.html
        log_info "Docs written to docs/"
    }

    # --- NEW: task::run ---
    task_run() {
        depends_on "build"
        log_info "Starting app (simulated 2s)..."
        sleep 2
        log_info "App started and stopped."
    }

    # Test: run docs
    echo "--- Running task: docs ---"
    TASK_EXECUTED=()
    execute_task "docs"

    echo ""

    # Test: run 'run'
    echo "--- Running task: run ---"
    TASK_EXECUTED=()
    execute_task "run"

    echo ""
    echo "--- Verification ---"
    echo "  Both tasks resolved dependencies: clean -> deps -> build -> docs/run"

    rm -rf /tmp/taskrunner_build_$$
}

# === Exercise 2: Detect Circular Dependencies ===
# Problem: Add circular dependency detection using a TASK_VISITING array.
exercise_2() {
    echo "=== Exercise 2: Detect Circular Dependencies ==="

    declare -A TASK_DEPS
    declare -A TASK_EXECUTED
    declare -A TASK_VISITING
    local visit_stack=()

    log_info()    { echo "  [INFO] $*"; }
    log_error()   { echo "  [ERROR] $*" >&2; }

    execute_task() {
        local task="$1"

        # Skip if already completed
        if [ -n "${TASK_EXECUTED[$task]:-}" ]; then
            return 0
        fi

        # Check for circular dependency
        if [ -n "${TASK_VISITING[$task]:-}" ]; then
            # Build the cycle path for error message
            local cycle="${visit_stack[*]} -> $task"
            log_error "Circular dependency detected: $cycle"
            return 1
        fi

        # Mark as visiting
        TASK_VISITING["$task"]=1
        visit_stack+=("$task")

        # Execute dependencies
        local deps="${TASK_DEPS[$task]:-}"
        for dep in $deps; do
            if ! execute_task "$dep"; then
                return 1
            fi
        done

        # Execute this task
        log_info "Executing: $task"

        # Mark as done, remove from visiting
        TASK_EXECUTED["$task"]=1
        unset TASK_VISITING["$task"]
        # Remove from visit stack
        unset 'visit_stack[${#visit_stack[@]}-1]'

        return 0
    }

    # Test 1: Normal dependencies (no cycle)
    echo "--- Test 1: Normal dependencies ---"
    TASK_DEPS=([build]="clean deps" [test]="build" [deploy]="test")
    TASK_EXECUTED=()
    TASK_VISITING=()
    visit_stack=()

    if execute_task "deploy"; then
        echo "  Result: SUCCESS (deploy ran clean -> deps -> build -> test -> deploy)"
    else
        echo "  Result: FAILED"
    fi

    echo ""

    # Test 2: Circular dependency (build -> test -> build)
    echo "--- Test 2: Circular dependency ---"
    TASK_DEPS=([build]="test" [test]="build")
    TASK_EXECUTED=()
    TASK_VISITING=()
    visit_stack=()

    if execute_task "build" 2>&1; then
        echo "  Result: UNEXPECTED SUCCESS"
    else
        echo "  Result: Correctly detected cycle and aborted"
    fi

    echo ""

    # Test 3: Deeper cycle (A -> B -> C -> A)
    echo "--- Test 3: Deeper cycle (A -> B -> C -> A) ---"
    TASK_DEPS=([A]="B" [B]="C" [C]="A")
    TASK_EXECUTED=()
    TASK_VISITING=()
    visit_stack=()

    if execute_task "A" 2>&1; then
        echo "  Result: UNEXPECTED SUCCESS"
    else
        echo "  Result: Correctly detected cycle and aborted"
    fi
}

# === Exercise 3: Implement the --dry-run Flag ===
# Problem: --dry-run prints what would execute without running tasks.
exercise_3() {
    echo "=== Exercise 3: Implement the --dry-run Flag ==="

    local DRY_RUN=0
    declare -A TASK_DEPS
    declare -A TASK_EXECUTED

    log_info()  { echo "  [INFO] $*"; }
    log_dry()   { echo "  [DRY RUN] Would execute: $*"; }

    execute_task() {
        local task="$1"

        if [ -n "${TASK_EXECUTED[$task]:-}" ]; then
            return 0
        fi

        # Execute dependencies first
        local deps="${TASK_DEPS[$task]:-}"
        for dep in $deps; do
            execute_task "$dep"
        done

        # Dry run or real execution
        if (( DRY_RUN )); then
            log_dry "$task"
        else
            log_info "Executing: $task"
            # Real task work would go here
        fi

        TASK_EXECUTED["$task"]=1
    }

    # Setup dependency graph:
    # deploy -> package -> test -> build -> deps -> clean
    TASK_DEPS=(
        [clean]=""
        [deps]="clean"
        [build]="deps"
        [test]="build"
        [package]="test"
        [deploy]="package"
    )

    # Test 1: Dry run of deploy
    echo "--- Test 1: --dry-run deploy ---"
    DRY_RUN=1
    TASK_EXECUTED=()
    execute_task "deploy"

    echo ""

    # Test 2: Normal execution of deploy
    echo "--- Test 2: Normal execution of deploy ---"
    DRY_RUN=0
    TASK_EXECUTED=()
    execute_task "deploy"

    echo ""

    # Test 3: Dry run shows deduplication
    echo "--- Test 3: --dry-run build test (build appears in both chains) ---"
    DRY_RUN=1
    TASK_EXECUTED=()
    execute_task "build"
    execute_task "test"
    echo "  (Note: build dependencies only shown once due to deduplication)"
}

# === Exercise 4: Implement Task Timing ===
# Problem: Track per-task execution time in milliseconds.
exercise_4() {
    echo "=== Exercise 4: Implement Task Timing ==="

    declare -A TASK_DEPS
    declare -A TASK_EXECUTED

    execute_task() {
        local task="$1"

        if [ -n "${TASK_EXECUTED[$task]:-}" ]; then
            return 0
        fi

        # Execute dependencies first
        local deps="${TASK_DEPS[$task]:-}"
        for dep in $deps; do
            execute_task "$dep"
        done

        # Record start time
        local start_ns
        # date +%s%N works on Linux but not all macOS
        # Fall back to seconds-only if needed
        if date +%s%N &>/dev/null && [[ "$(date +%s%N)" != *N ]]; then
            start_ns=$(date +%s%N)
        else
            start_ns=$(($(date +%s) * 1000000000))
        fi

        echo -n "  [$(date +%H:%M:%S)] Running task: $task..."

        # Simulate actual task work
        case "$task" in
            clean)   sleep 0.1 ;;
            deps)    sleep 0.2 ;;
            build)   sleep 0.3 ;;
            test)    sleep 0.2 ;;
            package) sleep 0.1 ;;
            *)       sleep 0.1 ;;
        esac

        # Record end time and calculate duration
        local end_ns
        if date +%s%N &>/dev/null && [[ "$(date +%s%N)" != *N ]]; then
            end_ns=$(date +%s%N)
        else
            end_ns=$(($(date +%s) * 1000000000))
        fi

        local duration_ms=$(( (end_ns - start_ns) / 1000000 ))

        echo " done (${duration_ms}ms)"

        TASK_EXECUTED["$task"]=1
    }

    # Setup
    TASK_DEPS=(
        [clean]=""
        [deps]="clean"
        [build]="deps"
        [test]="build"
        [package]="test"
    )

    echo "--- Running task: package (with timing) ---"
    local total_start
    if date +%s%N &>/dev/null && [[ "$(date +%s%N)" != *N ]]; then
        total_start=$(date +%s%N)
    else
        total_start=$(($(date +%s) * 1000000000))
    fi

    execute_task "package"

    local total_end
    if date +%s%N &>/dev/null && [[ "$(date +%s%N)" != *N ]]; then
        total_end=$(date +%s%N)
    else
        total_end=$(($(date +%s) * 1000000000))
    fi
    local total_ms=$(( (total_end - total_start) / 1000000 ))

    echo ""
    echo "  Total time: ${total_ms}ms"
}

# === Exercise 5: Write Bats Tests for the Task Runner ===
# Problem: Test --list, unknown task, dependency execution, deduplication.
exercise_5() {
    echo "=== Exercise 5: Write Bats Tests for the Task Runner ==="

    local work_dir="/tmp/taskrunner_bats_$$"
    mkdir -p "$work_dir"

    # Create a minimal task runner script for testing
    cat > "$work_dir/task.sh" << 'TASKSH'
#!/bin/bash
set -euo pipefail

declare -A TASK_DEPS
declare -A TASK_EXECUTED
EXEC_LOG=""

log_info()    { echo "[INFO] $*"; }
log_error()   { echo "[ERROR] $*" >&2; }
log_success() { echo "[OK] $*"; }

task_clean() { EXEC_LOG="${EXEC_LOG}clean,"; }
task_deps()  { depends_on "clean"; EXEC_LOG="${EXEC_LOG}deps,"; }
task_build() { depends_on "deps"; EXEC_LOG="${EXEC_LOG}build,"; }
task_test()  { depends_on "build"; EXEC_LOG="${EXEC_LOG}test,"; }

TASKS_FOUND=(clean deps build test)

depends_on() {
    for dep in "$@"; do
        execute_task "$dep"
    done
}

execute_task() {
    local task="$1"
    [ -n "${TASK_EXECUTED[$task]:-}" ] && return 0
    log_info "Running task: $task"
    "task_$task"
    TASK_EXECUTED["$task"]=1
    log_success "Task '$task' completed"
}

list_tasks() {
    for t in "${TASKS_FOUND[@]}"; do
        echo "$t"
    done
}

main() {
    case "${1:-}" in
        --list|-l)
            list_tasks
            exit 0
            ;;
    esac

    for task in "$@"; do
        if [[ ! " ${TASKS_FOUND[*]} " =~ " ${task} " ]]; then
            log_error "Unknown task: $task"
            exit 1
        fi
    done

    for task in "$@"; do
        execute_task "$task"
    done

    echo "EXEC_ORDER=${EXEC_LOG}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
TASKSH
    chmod +x "$work_dir/task.sh"

    # Create Bats test file
    cat > "$work_dir/test_task_runner.bats" << 'BATS'
#!/usr/bin/env bats

setup() {
    TASK_SH="${BATS_TEST_DIRNAME}/task.sh"
}

@test "--list outputs at least clean, build, and test" {
    run bash "$TASK_SH" --list
    [ "$status" -eq 0 ]
    [[ "$output" =~ "clean" ]]
    [[ "$output" =~ "build" ]]
    [[ "$output" =~ "test" ]]
}

@test "unknown_task exits with non-zero code and prints error" {
    run bash "$TASK_SH" unknown_task
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Unknown task" ]]
}

@test "build also runs clean and deps (dependencies execute)" {
    run bash "$TASK_SH" build
    [ "$status" -eq 0 ]
    [[ "$output" =~ "Running task: clean" ]]
    [[ "$output" =~ "Running task: deps" ]]
    [[ "$output" =~ "Running task: build" ]]
}

@test "build build (same task twice) only executes build once" {
    run bash "$TASK_SH" build build
    [ "$status" -eq 0 ]
    # EXEC_ORDER should have build only once
    [[ "$output" =~ "EXEC_ORDER=clean,deps,build," ]]
}
BATS

    echo "--- task.sh (minimal version) ---"
    head -15 "$work_dir/task.sh" | sed 's/^/  /'
    echo "  ..."

    echo ""
    echo "--- test_task_runner.bats ---"
    cat "$work_dir/test_task_runner.bats" | sed 's/^/  /'

    echo ""
    echo "--- Running tests manually ---"

    local pass=0 fail=0

    # Test 1: --list
    output=$(bash "$work_dir/task.sh" --list 2>&1)
    if echo "$output" | grep -q "clean" && echo "$output" | grep -q "build" && echo "$output" | grep -q "test"; then
        echo "  ok 1 - --list outputs clean, build, and test"
        (( pass++ ))
    else
        echo "  not ok 1 - --list missing expected tasks"
        (( fail++ ))
    fi

    # Test 2: unknown task
    output=$(bash "$work_dir/task.sh" unknown_task 2>&1)
    status=$?
    if [ $status -ne 0 ] && echo "$output" | grep -q "Unknown task"; then
        echo "  ok 2 - unknown_task exits non-zero with error"
        (( pass++ ))
    else
        echo "  not ok 2 - unknown_task handling failed"
        (( fail++ ))
    fi

    # Test 3: build runs dependencies
    output=$(bash "$work_dir/task.sh" build 2>&1)
    if echo "$output" | grep -q "Running task: clean" && \
       echo "$output" | grep -q "Running task: deps" && \
       echo "$output" | grep -q "Running task: build"; then
        echo "  ok 3 - build runs clean and deps"
        (( pass++ ))
    else
        echo "  not ok 3 - build dependencies not resolved"
        (( fail++ ))
    fi

    # Test 4: deduplication
    output=$(bash "$work_dir/task.sh" build build 2>&1)
    exec_order=$(echo "$output" | grep "EXEC_ORDER=" | sed 's/.*EXEC_ORDER=//')
    if [ "$exec_order" = "clean,deps,build," ]; then
        echo "  ok 4 - build build only executes build once"
        (( pass++ ))
    else
        echo "  not ok 4 - deduplication failed (got: $exec_order)"
        (( fail++ ))
    fi

    echo ""
    echo "  $pass passed, $fail failed"

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
