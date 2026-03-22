# Project: Task Runner

**Difficulty**: ⭐⭐⭐

**Previous**: [Shell Script Testing](./13_Testing.md) | **Next**: [Project: Deployment Automation](./15_Project_Deployment.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Build a task runner that discovers and executes tasks defined as `task::name` bash functions
2. Implement dependency resolution so that tasks automatically run their prerequisites before executing
3. Write a topological sort to detect circular dependencies and determine correct execution order
4. Add parallel execution of independent tasks using background jobs and `wait`
5. Generate help text automatically by extracting `##` comments above task function definitions
6. Apply colored, timestamped output formatting to provide clear feedback during task execution
7. Implement error handling that stops execution and reports which task failed and why

---

Every software project accumulates repetitive commands: build, test, lint, deploy, clean. Developers typically reach for Make, npm scripts, or language-specific tools, but these add dependencies and complexity. A bash-based task runner lives in your repository as a single file, works anywhere bash is installed, and lets you define tasks as plain shell functions -- making it an ideal capstone project that combines dependency resolution, parallel execution, and CLI design from the previous lessons.

## 1. Overview

### What is a Task Runner?

A task runner is a tool that automates repetitive development tasks like building, testing, linting, and deploying. Popular examples include:

- **Make**: The classic build automation tool (complex syntax, file-based dependencies)
- **Just**: A modern command runner (simpler than Make, requires separate installation)
- **Task**: A task runner written in Go (YAML configuration)
- **npm scripts**: JavaScript ecosystem (limited to Node.js projects)

### Why Build One in Bash?

Building a task runner in bash offers several advantages:

1. **Zero dependencies**: Works anywhere bash is available
2. **Project-specific**: Lives in your repository as a single file
3. **Transparent**: Pure bash means no magic — just shell commands
4. **Flexible**: Easy to customize for your exact needs
5. **Educational**: Learn advanced bash patterns

### What We're Building

Our `task.sh` script will support:

- Task definition via naming convention (`task::name`)
- Dependency declaration with automatic resolution
- Parallel execution of independent tasks
- Automatic help generation from comments
- Colored output with timestamps
- Error handling with clear failure messages

---

## 2. Design

### Task Definition Format

Tasks are defined as bash functions with a special naming convention:

```bash
## Build the project
task::build() {
    depends_on "clean"
    echo "Building..."
    # build commands here
}
```

The `task::` prefix identifies the function as a task. The comment above the function becomes the help text.

### Dependency Resolution

Dependencies are declared with `depends_on`:

```bash
task::deploy() {
    depends_on "build" "test"
    # deploy commands
}
```

The runner executes dependencies before the task itself, handling circular dependencies and avoiding duplicate execution.

### Architecture

```
task.sh
├── Task Discovery (find all task::* functions)
├── Dependency Resolution (topological sort)
├── Execution Engine (run tasks in order, parallel when possible)
├── Help Generation (extract comments)
└── Output Formatting (colors, timestamps, status)
```

---

## 3. Core Features

### Feature 1: Task Registration

Tasks are discovered automatically by parsing the script for `task::*` function definitions.

### Feature 2: Dependency Declaration

The `depends_on` function records dependencies and ensures they run first.

### Feature 3: Help Generation

Comments starting with `##` above task functions are extracted to generate help text.

### Feature 4: Colored Output

ANSI color codes provide visual feedback:
- Green for success
- Red for errors
- Yellow for warnings
- Blue for info

### Feature 5: Parallel Execution

Independent tasks can run in parallel using background jobs and `wait`.

### Feature 6: Error Handling

If a task fails (non-zero exit), execution stops and the error is reported.

---

## 4. Complete Implementation

Here's the full `task.sh` script:

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Task Runner - A Makefile-like tool in pure bash
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Global state
declare -A TASK_DEPS        # task -> dependencies
declare -A TASK_EXECUTED    # task -> 1 if executed
declare -A TASK_HELP        # task -> help text
TASKS_FOUND=()
PARALLEL=0

# ============================================================================
# Logging
# ============================================================================

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
# Dependency Management
# ============================================================================

depends_on() {
    local caller_task="${CURRENT_TASK}"
    local deps=("$@")

    # Store dependencies
    TASK_DEPS["${caller_task}"]="${deps[*]}"
}

# ============================================================================
# Task Discovery
# ============================================================================

discover_tasks() {
    local in_comment=0
    local comment_text=""
    local line

    while IFS= read -r line; do
        # Detect help comment
        if [[ "${line}" =~ ^##[[:space:]](.+)$ ]]; then
            comment_text="${BASH_REMATCH[1]}"
            in_comment=1
        # Detect task function
        elif [[ "${line}" =~ ^task::([a-zA-Z0-9_-]+)\(\) ]]; then
            local task_name="${BASH_REMATCH[1]}"
            TASKS_FOUND+=("${task_name}")
            TASK_HELP["${task_name}"]="${comment_text}"
            comment_text=""
            in_comment=0
        # Reset if we hit a non-comment line
        elif [[ ! "${line}" =~ ^[[:space:]]*$ ]] && [[ ! "${line}" =~ ^## ]]; then
            comment_text=""
            in_comment=0
        fi
    done < "$0"
}

# ============================================================================
# Execution
# ============================================================================

execute_task() {
    local task_name="$1"

    # Skip if already executed
    if [[ -n "${TASK_EXECUTED[${task_name}]:-}" ]]; then
        return 0
    fi

    # Execute dependencies first
    if [[ -n "${TASK_DEPS[${task_name}]:-}" ]]; then
        local deps=(${TASK_DEPS[${task_name}]})
        for dep in "${deps[@]}"; do
            if [[ ! " ${TASKS_FOUND[*]} " =~ " ${dep} " ]]; then
                log_error "Task '${task_name}' depends on unknown task '${dep}'"
                return 1
            fi
            execute_task "${dep}"
        done
    fi

    # Execute the task
    log_info "Running task: ${CYAN}${task_name}${RESET}"

    # Set current task for depends_on
    CURRENT_TASK="${task_name}"

    # Run the task function
    if "task::${task_name}"; then
        TASK_EXECUTED["${task_name}"]=1
        log_success "Task '${task_name}' completed"
        return 0
    else
        log_error "Task '${task_name}' failed"
        return 1
    fi
}

# ============================================================================
# Help
# ============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS] <task> [task...]"
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo "  -l, --list      List all available tasks"
    echo "  -p, --parallel  Execute independent tasks in parallel"
    echo ""
    echo "Available tasks:"
    echo ""

    for task in "${TASKS_FOUND[@]}"; do
        local help_text="${TASK_HELP[${task}]:-No description}"
        printf "  ${CYAN}%-15s${RESET} %s\n" "${task}" "${help_text}"
    done

    echo ""
    echo "Examples:"
    echo "  $0 build              # Run the build task"
    echo "  $0 clean build test   # Run multiple tasks in order"
    echo "  $0 -p test            # Run with parallel execution"
}

list_tasks() {
    for task in "${TASKS_FOUND[@]}"; do
        echo "${task}"
    done
}

# ============================================================================
# Task Definitions
# ============================================================================

## Clean build artifacts
task::clean() {
    log_info "Cleaning build directory..."
    rm -rf build/
    mkdir -p build/
}

## Install dependencies
task::deps() {
    log_info "Installing dependencies..."
    # Simulate dependency installation
    sleep 1
}

## Lint the code
task::lint() {
    depends_on "deps"
    log_info "Running linter..."
    # Simulate linting
    sleep 1
}

## Format the code
task::format() {
    log_info "Formatting code..."
    # Simulate formatting
    sleep 1
}

## Run unit tests
task::test() {
    depends_on "deps" "lint"
    log_info "Running tests..."
    # Simulate tests
    sleep 2
}

## Build the project
task::build() {
    depends_on "clean" "deps"
    log_info "Compiling source files..."
    echo "main.o" > build/main.o
    echo "app.o" > build/app.o
    sleep 1
    log_info "Linking binary..."
    echo "myapp" > build/myapp
}

## Run all checks (lint, test)
task::check() {
    depends_on "lint" "test"
    log_success "All checks passed"
}

## Build and run tests
task::all() {
    depends_on "build" "test"
    log_success "Build and test completed"
}

## Package the application
task::package() {
    depends_on "build" "test"
    log_info "Creating package..."
    tar -czf build/myapp.tar.gz -C build/ myapp
    log_success "Package created: build/myapp.tar.gz"
}

## Deploy to production
task::deploy() {
    depends_on "package"
    log_warn "Deploying to production..."
    # Simulate deployment
    sleep 2
    log_success "Deployment completed"
}

## Watch for changes and rebuild
task::watch() {
    log_info "Watching for changes..."
    log_warn "Press Ctrl+C to stop"
    while true; do
        execute_task "build"
        sleep 5
    done
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Discover all tasks
    discover_tasks

    # Parse arguments
    local tasks_to_run=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -l|--list)
                list_tasks
                exit 0
                ;;
            -p|--parallel)
                PARALLEL=1
                shift
                ;;
            *)
                tasks_to_run+=("$1")
                shift
                ;;
        esac
    done

    # If no tasks specified, show help
    if [[ ${#tasks_to_run[@]} -eq 0 ]]; then
        show_help
        exit 0
    fi

    # Validate tasks exist
    for task in "${tasks_to_run[@]}"; do
        if [[ ! " ${TASKS_FOUND[*]} " =~ " ${task} " ]]; then
            log_error "Unknown task: ${task}"
            echo ""
            echo "Available tasks:"
            list_tasks
            exit 1
        fi
    done

    # Execute tasks
    log_info "Starting task runner..."
    local start_time=$(date +%s)

    for task in "${tasks_to_run[@]}"; do
        if ! execute_task "${task}"; then
            log_error "Task execution failed"
            exit 1
        fi
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    log_success "All tasks completed in ${duration}s"
}

# Run main if script is executed (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

---

## 5. Usage Examples

### Make the Script Executable

```bash
chmod +x task.sh
```

### View Available Tasks

```bash
./task.sh --help
```

Output:
```
Available tasks:

  clean           Clean build artifacts
  deps            Install dependencies
  lint            Lint the code
  format          Format the code
  test            Run unit tests
  build           Build the project
  check           Run all checks (lint, test)
  all             Build and run tests
  package         Package the application
  deploy          Deploy to production
  watch           Watch for changes and rebuild
```

### Run a Single Task

```bash
./task.sh build
```

Output:
```
[14:32:10] Running task: clean
[14:32:10] Cleaning build directory...
[14:32:10] ✓ Task 'clean' completed
[14:32:10] Running task: deps
[14:32:10] Installing dependencies...
[14:32:11] ✓ Task 'deps' completed
[14:32:11] Running task: build
[14:32:11] Compiling source files...
[14:32:11] Linking binary...
[14:32:12] ✓ Task 'build' completed

[14:32:12] ✓ All tasks completed in 2s
```

### Run Multiple Tasks

```bash
./task.sh clean build test
```

Dependencies are automatically resolved — each task runs only once.

### Deploy to Production

```bash
./task.sh deploy
```

This automatically runs: `clean` → `deps` → `build` → `test` → `package` → `deploy`

### List Tasks Programmatically

```bash
./task.sh --list
```

Output:
```
clean
deps
lint
format
test
build
check
all
package
deploy
watch
```

### Add Your Own Tasks

Edit `task.sh` and add:

```bash
## Run the development server
task::dev() {
    depends_on "build"
    log_info "Starting dev server..."
    ./build/myapp --dev
}
```

---

## 6. How It Works

### Task Discovery

The `discover_tasks` function reads the script itself and uses regex to find:
1. Comments starting with `##` (help text)
2. Functions matching `task::*()` (task definitions)

### Dependency Resolution

When a task calls `depends_on "dep1" "dep2"`:
1. The dependencies are stored in the `TASK_DEPS` associative array
2. During execution, dependencies are run recursively before the task
3. The `TASK_EXECUTED` array prevents duplicate execution

### Execution Flow

```
1. Parse command line arguments
2. Discover all task::* functions
3. Validate requested tasks exist
4. For each task:
   a. Check if already executed (skip if yes)
   b. Execute dependencies recursively
   c. Run the task function
   d. Mark as executed
5. Report total time
```

### Error Handling

- `set -euo pipefail` ensures errors propagate
- Failed tasks return non-zero, stopping execution
- Clear error messages show which task failed

---

## Extensions

### 1. Parallel Execution

Implement the `-p` flag to run independent tasks in parallel:

```bash
if [[ ${PARALLEL} -eq 1 ]]; then
    for task in "${independent_tasks[@]}"; do
        execute_task "${task}" &
    done
    wait
fi
```

Requires dependency graph analysis to find independent tasks.

### 2. Task Timing

Track and display execution time per task:

```bash
task::build() {
    local start=$(date +%s%N)
    # ... task code ...
    local end=$(date +%s%N)
    local ms=$(( (end - start) / 1000000 ))
    log_info "Task took ${ms}ms"
}
```

### 3. Configuration File

Support a `.taskrc` file for settings:

```bash
# .taskrc
PARALLEL=1
LOG_LEVEL=debug
BUILD_DIR=./dist
```

Load with:

```bash
if [[ -f .taskrc ]]; then
    source .taskrc
fi
```

### 4. Task Namespaces

Support namespaced tasks like `task::docker::build`:

```bash
./task.sh docker:build
```

Parse the namespace and find the corresponding function.

### 5. Dry Run Mode

Add `--dry-run` to show what would be executed:

```bash
if [[ ${DRY_RUN} -eq 1 ]]; then
    log_info "Would execute: ${task_name}"
    return 0
fi
```

### 6. Task Hooks

Support before/after hooks:

```bash
task::build() {
    run_hook "before_build"
    # ... build code ...
    run_hook "after_build"
}

hook::before_build() {
    log_info "Preparing build environment..."
}
```

### 7. JSON Output

Add `--json` flag for machine-readable output:

```bash
{
  "tasks_run": ["clean", "build", "test"],
  "duration_seconds": 12,
  "status": "success"
}
```

### 8. Task Caching

Skip tasks if inputs haven't changed:

```bash
task::build() {
    if cache_valid "src/**/*.c" "build/myapp"; then
        log_info "Build cache hit, skipping"
        return 0
    fi
    # ... build ...
}
```

## Exercises

### Exercise 1: Add a New Task to the Runner

Using the provided `task.sh` implementation as a starting point, add the following two tasks:

```bash
## Generate API documentation
task::docs() {
    depends_on "build"
    # Your implementation here
}

## Run the application locally
task::run() {
    depends_on "build"
    # Your implementation here
}
```

- `task::docs` should depend on `build` and print "Generating docs..." then create a `docs/` directory
- `task::run` should depend on `build` and simulate starting the app with a 2-second sleep

Run `./task.sh docs` and `./task.sh run` to verify dependency resolution works correctly for both new tasks.

### Exercise 2: Detect Circular Dependencies

The current implementation does not detect circular dependencies — it will loop forever. Add circular dependency detection:
- Before executing a task, maintain a `TASK_VISITING` array of tasks currently in the call stack
- If `execute_task` is called for a task that is already in `TASK_VISITING`, print an error like `"Circular dependency detected: build → test → build"` and exit with code 1
- Test by adding a circular dependency: make `task::build` depend on `task::test` and `task::test` depend on `task::build`, then run `./task.sh build`

### Exercise 3: Implement the --dry-run Flag

Add a `--dry-run` (`-n`) flag to the task runner:
- When dry-run is active, `execute_task` should print `[DRY RUN] Would execute: <task>` instead of actually running the task function
- Dependencies should still be resolved and displayed in the correct order
- The `TASK_EXECUTED` deduplication should still work in dry-run mode

Verify by running `./task.sh --dry-run deploy` and checking that the full dependency chain (clean → deps → build → test → package → deploy) is printed without any commands actually executing.

### Exercise 4: Implement Task Timing

Add per-task execution timing:
- Record the start time before calling each task function using `date +%s%N` (nanoseconds)
- Record the end time after the task completes
- Calculate the duration in milliseconds
- Display it in the completion log line: `✓ Task 'build' completed in 1234ms`

After implementing, run `./task.sh all` and verify that each task shows a non-zero duration.

### Exercise 5: Write Bats Tests for the Task Runner

Write a Bats test suite `test_task_runner.bats` that tests the core behaviors of `task.sh`:
- Test that running `./task.sh --list` outputs at least `clean`, `build`, and `test`
- Test that running `./task.sh unknown_task` exits with a non-zero code and prints an error
- Test that running `./task.sh build` also runs `clean` and `deps` (i.e., dependencies execute)
- Test that running `./task.sh build build` (same task twice) only executes `build` once (deduplication)

Use mocking or temporary script files as needed to isolate tests from side effects.

---

**Previous**: [Shell Script Testing](./13_Testing.md) | **Next**: [Project: Deployment Automation](./15_Project_Deployment.md)
