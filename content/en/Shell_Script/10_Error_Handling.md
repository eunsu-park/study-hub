# Lesson 10: Error Handling and Debugging

**Difficulty**: ⭐⭐⭐

**Previous**: [Process Management and Job Control](./09_Process_Management.md) | **Next**: [Argument Parsing and CLI Interfaces](./11_Argument_Parsing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the behavior and edge cases of `set -e`, `set -u`, and `set -o pipefail`
2. Configure the recommended strict-mode header (`set -euo pipefail`) for production scripts
3. Implement `trap ERR` handlers that capture exit codes, line numbers, and stack traces
4. Build a reusable error handling framework with named exit codes, log levels, and die/assert functions
5. Apply defensive coding patterns including input validation, safe temporary files, and lock file management
6. Use ShellCheck to detect and fix common shell scripting bugs before runtime
7. Write multi-level debug logging with configurable verbosity and structured key-value output
8. Distinguish between `ERR` and `EXIT` traps and combine them for robust cleanup and error reporting

---

A shell script without error handling will silently continue after failures, corrupt data, and leave behind orphaned resources. In production environments -- where scripts manage backups, deployments, and data pipelines -- a single unhandled error can cascade into hours of downtime. This lesson teaches you how to make scripts fail fast, fail loudly, and clean up after themselves using the defensive patterns that separate throwaway scripts from reliable automation.

## 1. set Options Deep Dive

The `set` command controls shell behavior and error handling. Understanding these options is crucial for writing robust scripts.

### set -e (errexit)

```bash
#!/bin/bash

# Exit immediately if any command returns non-zero
set -e

echo "Starting..."
false  # This will cause the script to exit
echo "This won't be printed"
```

### Gotchas with set -e

```bash
#!/bin/bash
set -e

# set -e does NOT exit in these cases:

# 1. Commands in conditions
if false; then
    echo "Won't execute"
fi
echo "Still running"

# 2. Commands with || or &&
false || echo "This runs"
echo "Still running"

# 3. Commands in a pipeline (except the last one, unless pipefail is set)
false | echo "Pipeline continues"
echo "Still running"

# 4. Commands in functions called in conditions
check_something() {
    false  # Won't exit the script if called in condition
    return 1
}

if check_something; then
    echo "Won't execute"
fi
echo "Still running after function"

# 5. Negated commands
! false  # Won't exit
echo "Still running after negation"
```

### set -u (nounset)

```bash
#!/bin/bash

# Exit if accessing undefined variable
set -u

defined_var="hello"
echo "$defined_var"  # OK

# This will cause exit
# echo "$undefined_var"  # Error: undefined_var: unbound variable

# Safe way to check if variable is set
echo "${undefined_var:-default_value}"  # Prints: default_value

# Check if variable is set before using
if [ -n "${undefined_var+x}" ]; then
    echo "Variable is set: $undefined_var"
else
    echo "Variable is not set"
fi

# Another pattern: use empty string as default
value="${undefined_var:-}"
if [ -n "$value" ]; then
    echo "Value: $value"
else
    echo "Variable was undefined"
fi
```

### set -o pipefail

```bash
#!/bin/bash

# Without pipefail
echo "Without pipefail:"
false | echo "Pipeline output"
echo "Exit status: $?"  # 0 (from echo)

# With pipefail
set -o pipefail
echo -e "\nWith pipefail:"
false | echo "Pipeline output"
echo "Exit status: $?"  # 1 (from false)

# Practical example
set -e
set -o pipefail

# This will exit the script if grep finds nothing
cat /var/log/syslog | grep "error" | head -10

# PIPESTATUS array contains exit codes of all pipeline commands
cat file.txt | grep "pattern" | sort | uniq
echo "Pipeline status: ${PIPESTATUS[@]}"
# Prints something like: 0 1 0 0
# (cat succeeded, grep failed, sort and uniq succeeded)
```

### set Options Comparison

| Option | Description | Effect | When to Use |
|--------|-------------|--------|-------------|
| `set -e` | errexit | Exit on command failure | Production scripts |
| `set -u` | nounset | Exit on undefined variable | Catch typos early |
| `set -o pipefail` | pipefail | Pipeline fails if any command fails | With `set -e` |
| `set -x` | xtrace | Print commands before execution | Debugging |
| `set -v` | verbose | Print shell input lines | Deep debugging |
| `set -n` | noexec | Read commands but don't execute | Syntax checking |
| `set -C` | noclobber | Prevent output redirection from overwriting | Protect files |

### Recommended Script Header

```bash
#!/bin/bash

# Strict mode
set -euo pipefail
IFS=$'\n\t'

# Now script will:
# - Exit on error (set -e)
# - Exit on undefined variable (set -u)
# - Exit if any pipeline command fails (set -o pipefail)
# - Use safe IFS (newline and tab only)

echo "Script running in strict mode"
```

### Temporarily Disabling set -e

```bash
#!/bin/bash
set -e

# Method 1: Use || true
false || true  # Won't exit
echo "Still running"

# Method 2: Use explicit if
if command_that_might_fail; then
    echo "Success"
else
    echo "Failed, but handling it"
fi

# Method 3: Temporarily disable
set +e
command_that_might_fail
status=$?
set -e

if [ $status -ne 0 ]; then
    echo "Command failed with status $status"
fi

# Method 4: Use ! to negate (exit code becomes 0)
if ! command_that_might_fail; then
    echo "Command failed as expected"
fi
```

## 2. trap ERR

The `ERR` trap is triggered when a command returns a non-zero exit status (except in the same cases where `set -e` wouldn't exit).

### Basic ERR Trap

```bash
#!/bin/bash
set -e

error_handler() {
    echo "Error occurred in script"
}

trap error_handler ERR

echo "Starting..."
false  # Triggers ERR trap
echo "This won't be reached"
```

### Getting Error Context

```bash
#!/bin/bash
set -e

error_handler() {
    local exit_code=$?
    local line_num=$1

    echo "========================================"
    echo "Error occurred!"
    echo "Exit code: $exit_code"
    echo "Line number: $line_num"
    echo "Command: $BASH_COMMAND"
    echo "========================================"

    # Exit with same code
    exit $exit_code
}

trap 'error_handler $LINENO' ERR

echo "Line 1"
echo "Line 2"
false  # Line 3 - will trigger error
echo "Line 4"
```

### Stack Trace Generation

```bash
#!/bin/bash
set -e

print_stack_trace() {
    local frame=0
    echo "Stack trace:"
    while caller $frame; do
        ((frame++))
    done | while read line func file; do
        echo "  at $func() in $file:$line"
    done
}

error_handler() {
    local exit_code=$?
    local line_num=$1

    echo "========================================"
    echo "ERROR: Command failed with exit code $exit_code"
    echo "  Line: $line_num"
    echo "  Command: $BASH_COMMAND"
    echo "========================================"

    print_stack_trace

    exit $exit_code
}

trap 'error_handler $LINENO' ERR

function level3() {
    echo "Level 3"
    false  # Error here
}

function level2() {
    echo "Level 2"
    level3
}

function level1() {
    echo "Level 1"
    level2
}

level1
```

### Advanced Error Handler

```bash
#!/bin/bash
set -euo pipefail

# Get detailed function stack
get_function_stack() {
    local i=0
    local stack=""

    while [ $i -lt ${#FUNCNAME[@]} ]; do
        local func="${FUNCNAME[$i]}"
        local line="${BASH_LINENO[$i-1]}"
        local src="${BASH_SOURCE[$i]}"

        # Skip the error handler itself
        if [ "$func" != "error_handler" ] && [ "$func" != "get_function_stack" ]; then
            stack="${stack}  → ${func}() at ${src}:${line}\n"
        fi

        ((i++))
    done

    echo -e "$stack"
}

error_handler() {
    local exit_code=$?
    local line_num="${BASH_LINENO[0]}"
    local src="${BASH_SOURCE[1]}"

    echo "╔════════════════════════════════════════════════════════════"
    echo "║ ERROR DETECTED"
    echo "╠════════════════════════════════════════════════════════════"
    echo "║ Exit Code    : $exit_code"
    echo "║ Failed Command: $BASH_COMMAND"
    echo "║ Location     : $src:$line_num"
    echo "╠════════════════════════════════════════════════════════════"
    echo "║ Call Stack:"
    echo "╠════════════════════════════════════════════════════════════"
    get_function_stack
    echo "╚════════════════════════════════════════════════════════════"

    exit $exit_code
}

trap 'error_handler' ERR

# Test it
function inner_function() {
    echo "Inner function executing..."
    nonexistent_command  # This will fail
}

function outer_function() {
    echo "Outer function executing..."
    inner_function
}

outer_function
```

### ERR vs EXIT Trap

```bash
#!/bin/bash
set -e

# ERR trap: only on error
trap 'echo "ERR trap: Command failed"' ERR

# EXIT trap: always on exit
trap 'echo "EXIT trap: Script exiting"' EXIT

echo "Normal execution"
# On normal exit, only EXIT trap runs

# Uncomment to see both traps:
# false
```

## 3. Custom Error Framework

Building a reusable error handling framework makes scripts more maintainable.

### Error Code Enum

```bash
#!/bin/bash

# Define error codes as readonly constants
readonly E_SUCCESS=0
readonly E_GENERAL=1
readonly E_MISUSE=2
readonly E_NOINPUT=66
readonly E_NOUSER=67
readonly E_NOHOST=68
readonly E_UNAVAILABLE=69
readonly E_SOFTWARE=70
readonly E_OSERR=71
readonly E_OSFILE=72
readonly E_CANTCREAT=73
readonly E_IOERR=74
readonly E_TEMPFAIL=75
readonly E_PROTOCOL=76
readonly E_NOPERM=77
readonly E_CONFIG=78

# Map codes to messages
declare -A ERROR_MESSAGES=(
    [1]="General error"
    [2]="Misuse of shell command"
    [66]="Input file missing or unreadable"
    [67]="User does not exist"
    [68]="Host does not exist"
    [69]="Service unavailable"
    [70]="Internal software error"
    [71]="System error"
    [72]="Critical OS file missing"
    [73]="Cannot create output file"
    [74]="I/O error"
    [75]="Temporary failure"
    [76]="Protocol error"
    [77]="Permission denied"
    [78]="Configuration error"
)

# Get error message for code
get_error_message() {
    local code=$1
    echo "${ERROR_MESSAGES[$code]:-Unknown error}"
}

# Usage
echo "Error 66: $(get_error_message 66)"
echo "Error 77: $(get_error_message 77)"
```

### Error Functions

```bash
#!/bin/bash
set -euo pipefail

# Color codes
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'  # No Color

# Error levels
readonly ERROR_LEVEL_INFO=0
readonly ERROR_LEVEL_WARN=1
readonly ERROR_LEVEL_ERROR=2
readonly ERROR_LEVEL_FATAL=3

# Log with level
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        $ERROR_LEVEL_INFO)
            echo "[$timestamp] INFO: $message" >&2
            ;;
        $ERROR_LEVEL_WARN)
            echo -e "[$timestamp] ${YELLOW}WARN${NC}: $message" >&2
            ;;
        $ERROR_LEVEL_ERROR)
            echo -e "[$timestamp] ${RED}ERROR${NC}: $message" >&2
            ;;
        $ERROR_LEVEL_FATAL)
            echo -e "[$timestamp] ${RED}FATAL${NC}: $message" >&2
            ;;
    esac
}

# Convenience functions
info() { log_message $ERROR_LEVEL_INFO "$*"; }
warn() { log_message $ERROR_LEVEL_WARN "$*"; }
error() { log_message $ERROR_LEVEL_ERROR "$*"; }
fatal() { log_message $ERROR_LEVEL_FATAL "$*"; exit 1; }

# Die function with exit code
die() {
    local code=$1
    shift
    error "$@"
    exit $code
}

# Assert function
assert() {
    local condition=$1
    shift
    local message=$*

    if ! eval "$condition"; then
        die 1 "Assertion failed: $message"
    fi
}

# Usage examples
info "Script starting..."
warn "This is a warning"
error "This is an error (but doesn't exit)"
assert "[ -f /etc/passwd ]" "/etc/passwd must exist"
assert "[ 1 -eq 1 ]" "Math still works"
# fatal "Critical error - exiting"  # Uncomment to test
info "Script completed"
```

### Try-Catch Simulation

```bash
#!/bin/bash

# Try-catch simulation using subshells
try() {
    # Execute commands in subshell
    # Return 0 if successful, 1 if any command fails
    ( eval "$*" )
    return $?
}

catch() {
    local exit_code=$1
    shift

    if [ $exit_code -ne 0 ]; then
        eval "$*"
        return 0
    fi

    return 1
}

# Usage
if try "echo 'Attempting operation'; false"; catch $? "echo 'Caught error!'"; then
    echo "Error was handled"
fi

# More complex example
perform_operation() {
    echo "Attempting risky operation..."

    # Simulate some work
    if [ $((RANDOM % 2)) -eq 0 ]; then
        echo "Operation succeeded"
        return 0
    else
        echo "Operation failed"
        return 1
    fi
}

if try "perform_operation"; catch $? "echo 'Operation failed, handling gracefully'"; then
    echo "Error was caught and handled"
else
    echo "Operation succeeded"
fi

# Alternative: using trap in subshell
try_with_trap() {
    (
        set -e
        trap 'return 1' ERR
        eval "$*"
    )
}

if try_with_trap "echo 'Working...'; false"; then
    echo "Success"
else
    echo "Caught error with trap method"
fi
```

### Complete Error Framework

```bash
#!/bin/bash
set -euo pipefail

# ============================================================================
# ERROR HANDLING FRAMEWORK
# ============================================================================

readonly SCRIPT_NAME=$(basename "$0")
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Exit codes
readonly E_SUCCESS=0
readonly E_GENERAL=1
readonly E_INVALID_ARGS=2
readonly E_FILE_NOT_FOUND=66
readonly E_PERMISSION_DENIED=77

# Colors
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly GREEN='\033[0;32m'
readonly NC='\033[0m'

# Log file
LOG_FILE="${SCRIPT_DIR}/${SCRIPT_NAME}.log"

# Initialize logging
init_logging() {
    exec 3>&1 4>&2  # Save stdout and stderr
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

# Restore file descriptors
cleanup_logging() {
    exec 1>&3 2>&4
    exec 3>&- 4>&-
}

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_info() {
    log "INFO: $*"
}

log_warn() {
    log "${YELLOW}WARN${NC}: $*" >&2
}

log_error() {
    log "${RED}ERROR${NC}: $*" >&2
}

log_success() {
    log "${GREEN}SUCCESS${NC}: $*"
}

# Error handler
error_handler() {
    local exit_code=$?
    local line_num="${BASH_LINENO[0]}"

    log_error "Command failed with exit code $exit_code at line $line_num"
    log_error "Failed command: $BASH_COMMAND"

    cleanup_logging
    exit $exit_code
}

# Setup traps
setup_traps() {
    trap error_handler ERR
    trap cleanup_logging EXIT
}

# Die function
die() {
    local code=$1
    shift
    log_error "$*"
    exit $code
}

# Check if command exists
require_command() {
    local cmd=$1
    if ! command -v "$cmd" &>/dev/null; then
        die $E_GENERAL "Required command not found: $cmd"
    fi
}

# Check if file exists
require_file() {
    local file=$1
    if [ ! -f "$file" ]; then
        die $E_FILE_NOT_FOUND "Required file not found: $file"
    fi
}

# Check if directory exists
require_directory() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        die $E_FILE_NOT_FOUND "Required directory not found: $dir"
    fi
}

# Validate number of arguments
validate_args() {
    local expected=$1
    local actual=$2

    if [ $actual -lt $expected ]; then
        die $E_INVALID_ARGS "Expected at least $expected arguments, got $actual"
    fi
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

main() {
    init_logging
    setup_traps

    log_info "Script started"

    # Validate requirements
    require_command "grep"
    require_command "sed"

    # Example operations
    log_info "Performing operations..."

    # This will succeed
    log_success "Operation completed successfully"

    # Uncomment to test error handling:
    # require_file "/nonexistent/file"
    # false

    log_info "Script completed successfully"
}

# Run main if not sourced
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
```

## 4. Defensive Coding Patterns

Defensive coding prevents errors before they happen.

### Input Validation

```bash
#!/bin/bash

# Validate string is not empty
validate_not_empty() {
    local var=$1
    local name=$2

    if [ -z "$var" ]; then
        echo "Error: $name cannot be empty" >&2
        return 1
    fi
}

# Validate number
validate_number() {
    local var=$1
    local name=$2

    if ! [[ "$var" =~ ^[0-9]+$ ]]; then
        echo "Error: $name must be a positive integer" >&2
        return 1
    fi
}

# Validate range
validate_range() {
    local var=$1
    local min=$2
    local max=$3
    local name=$4

    if [ "$var" -lt "$min" ] || [ "$var" -gt "$max" ]; then
        echo "Error: $name must be between $min and $max" >&2
        return 1
    fi
}

# Validate email
validate_email() {
    local email=$1

    if ! [[ "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        echo "Error: Invalid email address" >&2
        return 1
    fi
}

# Validate IP address
validate_ip() {
    local ip=$1

    if ! [[ "$ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
        echo "Error: Invalid IP address" >&2
        return 1
    fi

    # Check each octet is 0-255
    IFS='.' read -ra octets <<< "$ip"
    for octet in "${octets[@]}"; do
        if [ "$octet" -gt 255 ]; then
            echo "Error: Invalid IP address (octet > 255)" >&2
            return 1
        fi
    done
}

# Usage
username="john_doe"
validate_not_empty "$username" "username" || exit 1

age=25
validate_number "$age" "age" || exit 1
validate_range "$age" 0 150 "age" || exit 1

email="user@example.com"
validate_email "$email" || exit 1

ip="192.168.1.1"
validate_ip "$ip" || exit 1

echo "All validations passed"
```

### Checking Command Existence

```bash
#!/bin/bash

# Method 1: Using command -v
check_command_v() {
    local cmd=$1
    if command -v "$cmd" &>/dev/null; then
        echo "$cmd is available"
        return 0
    else
        echo "$cmd is not available" >&2
        return 1
    fi
}

# Method 2: Using type
check_command_type() {
    local cmd=$1
    if type "$cmd" &>/dev/null; then
        echo "$cmd is available"
        return 0
    else
        echo "$cmd is not available" >&2
        return 1
    fi
}

# Method 3: Using which (less portable)
check_command_which() {
    local cmd=$1
    if which "$cmd" &>/dev/null; then
        echo "$cmd is available"
        return 0
    else
        echo "$cmd is not available" >&2
        return 1
    fi
}

# Require command with helpful message
require_command() {
    local cmd=$1
    local install_hint=${2:-""}

    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: Required command '$cmd' not found" >&2
        if [ -n "$install_hint" ]; then
            echo "Install with: $install_hint" >&2
        fi
        exit 1
    fi
}

# Require one of multiple commands
require_one_of() {
    local found=0
    for cmd in "$@"; do
        if command -v "$cmd" &>/dev/null; then
            found=1
            break
        fi
    done

    if [ $found -eq 0 ]; then
        echo "Error: None of the required commands found: $*" >&2
        exit 1
    fi
}

# Usage
check_command_v "bash"
check_command_v "nonexistent_command" || echo "As expected"

require_command "grep"
require_command "curl" "apt-get install curl / brew install curl"

require_one_of "python3" "python"
require_one_of "vim" "nvim" "nano"

echo "All required commands are available"
```

### Safe Temporary Files

```bash
#!/bin/bash

# Create temp file
create_temp_file() {
    local tmpfile
    tmpfile=$(mktemp) || {
        echo "Failed to create temp file" >&2
        return 1
    }
    echo "$tmpfile"
}

# Create temp directory
create_temp_dir() {
    local tmpdir
    tmpdir=$(mktemp -d) || {
        echo "Failed to create temp directory" >&2
        return 1
    }
    echo "$tmpdir"
}

# Create temp file with custom template
create_temp_file_template() {
    local prefix=$1
    local tmpfile
    tmpfile=$(mktemp "/tmp/${prefix}.XXXXXX") || {
        echo "Failed to create temp file" >&2
        return 1
    }
    echo "$tmpfile"
}

# Safe temp file with cleanup
safe_temp_file() {
    local tmpfile
    tmpfile=$(mktemp) || return 1

    # Register cleanup
    trap "rm -f '$tmpfile'" EXIT

    echo "$tmpfile"
}

# Usage
TMPFILE=$(create_temp_file)
trap "rm -f '$TMPFILE'" EXIT

echo "data" > "$TMPFILE"
cat "$TMPFILE"

TMPDIR=$(create_temp_dir)
trap "rm -rf '$TMPDIR'" EXIT

echo "Created temp dir: $TMPDIR"
touch "$TMPDIR/file1.txt"
touch "$TMPDIR/file2.txt"
ls -la "$TMPDIR"

# Custom template
LOGFILE=$(create_temp_file_template "myapp_log")
echo "Log file: $LOGFILE"

echo "Cleanup will happen automatically on exit"
```

### Lock Files

```bash
#!/bin/bash

# Simple lock file
acquire_lock_simple() {
    local lockfile=$1

    if [ -e "$lockfile" ]; then
        echo "Lock file exists, another instance may be running" >&2
        return 1
    fi

    echo $$ > "$lockfile"
    trap "rm -f '$lockfile'" EXIT
    return 0
}

# Atomic lock with mkdir
acquire_lock_atomic() {
    local lockfile=$1
    local max_attempts=${2:-10}
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if mkdir "$lockfile" 2>/dev/null; then
            trap "rmdir '$lockfile'" EXIT
            return 0
        fi

        ((attempt++))
        echo "Lock attempt $attempt/$max_attempts failed, retrying..." >&2
        sleep 1
    done

    echo "Failed to acquire lock after $max_attempts attempts" >&2
    return 1
}

# Lock with PID check
acquire_lock_with_pid_check() {
    local lockfile=$1

    if [ -e "$lockfile" ]; then
        local pid=$(cat "$lockfile" 2>/dev/null)

        # Check if process is still running
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Another instance is running (PID: $pid)" >&2
            return 1
        else
            echo "Removing stale lock file" >&2
            rm -f "$lockfile"
        fi
    fi

    echo $$ > "$lockfile"
    trap "rm -f '$lockfile'" EXIT
    return 0
}

# Lock with flock (Linux)
acquire_lock_flock() {
    local lockfile=$1
    local fd=$2  # File descriptor to use

    # Open file descriptor
    eval "exec $fd>\"$lockfile\""

    # Try to acquire exclusive lock
    if flock -n "$fd"; then
        trap "flock -u '$fd'; exec $fd>&-; rm -f '$lockfile'" EXIT
        return 0
    else
        echo "Failed to acquire lock" >&2
        exec {fd}>&-
        return 1
    fi
}

# Usage
LOCKFILE="/tmp/myscript.lock"

if acquire_lock_with_pid_check "$LOCKFILE"; then
    echo "Lock acquired, doing work..."
    sleep 5
    echo "Work complete"
else
    echo "Could not acquire lock, exiting"
    exit 1
fi

# Alternative: using flock
# LOCKFILE="/tmp/myscript.flock"
# if acquire_lock_flock "$LOCKFILE" 200; then
#     echo "Flock acquired"
#     sleep 5
#     echo "Work complete"
# fi
```

### Safe File Operations

```bash
#!/bin/bash

# Safe file copy with verification
safe_copy() {
    local src=$1
    local dst=$2

    # Check source exists
    if [ ! -f "$src" ]; then
        echo "Error: Source file does not exist: $src" >&2
        return 1
    fi

    # Check destination doesn't exist or confirm overwrite
    if [ -e "$dst" ]; then
        echo "Warning: Destination exists: $dst" >&2
        read -p "Overwrite? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Copy cancelled" >&2
            return 1
        fi
    fi

    # Copy to temp file first
    local tmpfile="${dst}.tmp.$$"
    if ! cp "$src" "$tmpfile"; then
        echo "Error: Failed to copy file" >&2
        return 1
    fi

    # Verify copy
    if ! cmp -s "$src" "$tmpfile"; then
        echo "Error: Verification failed" >&2
        rm -f "$tmpfile"
        return 1
    fi

    # Move to final destination
    if ! mv "$tmpfile" "$dst"; then
        echo "Error: Failed to move temp file to destination" >&2
        rm -f "$tmpfile"
        return 1
    fi

    echo "Successfully copied $src to $dst"
    return 0
}

# Safe file write with backup
safe_write() {
    local file=$1
    local content=$2

    # Backup existing file
    if [ -f "$file" ]; then
        local backup="${file}.backup.$(date +%Y%m%d_%H%M%S)"
        if ! cp "$file" "$backup"; then
            echo "Error: Failed to create backup" >&2
            return 1
        fi
        echo "Created backup: $backup"
    fi

    # Write to temp file
    local tmpfile="${file}.tmp.$$"
    if ! echo "$content" > "$tmpfile"; then
        echo "Error: Failed to write to temp file" >&2
        return 1
    fi

    # Atomic move
    if ! mv "$tmpfile" "$file"; then
        echo "Error: Failed to move temp file" >&2
        rm -f "$tmpfile"
        return 1
    fi

    echo "Successfully wrote to $file"
    return 0
}

# Safe directory creation
safe_mkdir() {
    local dir=$1
    local mode=${2:-755}

    if [ -e "$dir" ]; then
        if [ ! -d "$dir" ]; then
            echo "Error: Path exists but is not a directory: $dir" >&2
            return 1
        fi
        echo "Directory already exists: $dir"
        return 0
    fi

    if ! mkdir -p -m "$mode" "$dir"; then
        echo "Error: Failed to create directory: $dir" >&2
        return 1
    fi

    echo "Created directory: $dir"
    return 0
}

# Usage
echo "test content" > /tmp/source.txt
safe_copy /tmp/source.txt /tmp/dest.txt
safe_write /tmp/output.txt "Hello, World!"
safe_mkdir /tmp/test_dir 755
```

## 5. ShellCheck Static Analysis

ShellCheck is a static analysis tool that finds bugs in shell scripts.

### Common ShellCheck Warnings

```bash
#!/bin/bash

# SC2086: Double quote to prevent word splitting
file="my file.txt"
cat $file        # BAD: will try to cat "my" and "file.txt"
cat "$file"      # GOOD: treats as single argument

# SC2046: Quote command substitution to prevent word splitting
for file in $(ls *.txt); do  # BAD
    echo "$file"
done

for file in *.txt; do        # GOOD
    echo "$file"
done

# SC2006: Use $(...) instead of `...`
result=`command`             # BAD (deprecated)
result=$(command)            # GOOD

# SC2155: Separate declaration and assignment to avoid masking return value
declare output=$(command)    # BAD: masks command's exit code
declare output               # GOOD
output=$(command)

# SC2164: Use || exit after cd in case it fails
cd /some/directory           # BAD: script continues if cd fails
cd /some/directory || exit   # GOOD

# SC2166: Prefer -a/-o over &&/|| in [ ] expressions
[ -f file && -r file ]       # BAD: doesn't work
[ -f file ] && [ -r file ]   # GOOD
[ -f file -a -r file ]       # GOOD (but [ ] is deprecated, use [[ ]])

# SC2006: Use [[ ]] instead of [ ] for better error handling
if [ $var = "value" ]; then  # BAD: fails if var is empty
    echo "match"
fi
if [[ $var = "value" ]]; then  # GOOD: handles empty var
    echo "match"
fi

# SC2143: Use grep -q instead of comparing output
if [ $(grep pattern file | wc -l) -gt 0 ]; then  # BAD
    echo "found"
fi
if grep -q pattern file; then  # GOOD
    echo "found"
fi

# SC2069: Redirecting stdout to stderr correctly
command 2>&1 >/dev/null      # BAD: wrong order
command >/dev/null 2>&1      # GOOD

# SC2181: Check exit code directly instead of $?
command
if [ $? -eq 0 ]; then        # BAD (acceptable but not ideal)
    echo "success"
fi
if command; then             # GOOD
    echo "success"
fi
```

### ShellCheck Integration

```bash
#!/bin/bash

# Install ShellCheck
# Ubuntu/Debian: apt-get install shellcheck
# macOS: brew install shellcheck
# Or download from: https://www.shellcheck.net/

# Check a script
shellcheck script.sh

# Check with specific severity
shellcheck --severity=warning script.sh

# Output in different formats
shellcheck --format=gcc script.sh    # GCC-style
shellcheck --format=json script.sh   # JSON format
shellcheck --format=tty script.sh    # Colored terminal output

# Exclude specific warnings
shellcheck --exclude=SC2086,SC2046 script.sh

# Check all scripts in directory
find . -name '*.sh' -exec shellcheck {} \;
```

### ShellCheck Configuration

```bash
#!/bin/bash

# .shellcheckrc file (place in project root or ~/.shellcheckrc)
# Disable specific checks globally
disable=SC2086,SC2046

# Set shell dialect
shell=bash

# Enable optional checks
enable=quote-safe-variables

# Example .shellcheckrc:
cat > .shellcheckrc << 'EOF'
# Disable word splitting warnings
disable=SC2086

# Enable all optional checks
enable=all

# Source paths
source-path=SCRIPTDIR
EOF
```

### Suppressing Warnings in Code

```bash
#!/bin/bash

# Suppress for next line
# shellcheck disable=SC2086
echo $unquoted_var

# Suppress for entire file
# shellcheck disable=SC2086,SC2046

# Suppress with explanation
# shellcheck disable=SC2086  # Intentional word splitting here
for word in $sentence; do
    echo "$word"
done

# Suppress for a block
# shellcheck disable=SC2086
{
    echo $var1
    echo $var2
    echo $var3
}
# shellcheck enable=SC2086
```

## 6. Debugging Techniques

Effective debugging techniques help identify and fix issues quickly.

### set -x (xtrace)

```bash
#!/bin/bash

# Enable xtrace (print commands before execution)
set -x

echo "This will be traced"
var="hello"
echo "$var"

# Disable xtrace
set +x

echo "This won't be traced"
```

### Custom PS4 for Better Tracing

```bash
#!/bin/bash

# Default PS4 is '+ '
# Customize it for more information

# Show line number
export PS4='+(${LINENO}): '
set -x
echo "Line number shown"
var="test"
echo "$var"
set +x

# Show line number and function name
export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
set -x

my_function() {
    echo "Inside function"
    local x=10
    echo "$x"
}

my_function
set +x

# Show timestamp, line, and function
export PS4='[$(date +%T)] ${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
set -x
echo "Timestamped trace"
set +x
```

### Selective Debugging

```bash
#!/bin/bash

# Debug flag
DEBUG=${DEBUG:-0}

debug() {
    if [ "$DEBUG" = "1" ]; then
        echo "DEBUG: $*" >&2
    fi
}

debug_start() {
    if [ "$DEBUG" = "1" ]; then
        set -x
    fi
}

debug_end() {
    if [ "$DEBUG" = "1" ]; then
        set +x
    fi
}

# Usage
debug "Script starting"

echo "Normal execution"

debug_start
# This section will be traced if DEBUG=1
var="test"
echo "$var"
result=$((var + 10))
debug_end

echo "More normal execution"

# Run with: DEBUG=1 ./script.sh
```

### set -v (verbose)

```bash
#!/bin/bash

# Verbose mode: print shell input lines
set -v

# This prints the line itself before executing
echo "Hello"
var="world"
echo "$var"

set +v
echo "Verbose mode off"
```

### Bash Debugger (bashdb)

```bash
#!/bin/bash

# Install bashdb
# Ubuntu/Debian: apt-get install bashdb
# Or download from: http://bashdb.sourceforge.net/

# Run script with debugger
# bashdb script.sh

# Debugger commands:
# n     - next line
# s     - step into function
# c     - continue until breakpoint
# l     - list source code
# p var - print variable value
# b N   - set breakpoint at line N
# q     - quit debugger

# Example script to debug
function calculate() {
    local a=$1
    local b=$2
    local result=$((a + b))
    echo "$result"
}

x=10
y=20
sum=$(calculate $x $y)
echo "Sum: $sum"

# Run with: bashdb this_script.sh
# Then use 'n' to step through, 'p x' to print variables, etc.
```

### Debug Logging

```bash
#!/bin/bash

# Debug levels
readonly DEBUG_NONE=0
readonly DEBUG_ERROR=1
readonly DEBUG_WARN=2
readonly DEBUG_INFO=3
readonly DEBUG_TRACE=4

DEBUG_LEVEL=${DEBUG_LEVEL:-$DEBUG_INFO}

debug_log() {
    local level=$1
    shift
    local message=$*

    if [ "$level" -le "$DEBUG_LEVEL" ]; then
        local level_name
        case $level in
            $DEBUG_ERROR) level_name="ERROR" ;;
            $DEBUG_WARN)  level_name="WARN"  ;;
            $DEBUG_INFO)  level_name="INFO"  ;;
            $DEBUG_TRACE) level_name="TRACE" ;;
        esac

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level_name] $message" >&2
    fi
}

error() { debug_log $DEBUG_ERROR "$*"; }
warn()  { debug_log $DEBUG_WARN "$*"; }
info()  { debug_log $DEBUG_INFO "$*"; }
trace() { debug_log $DEBUG_TRACE "$*"; }

# Usage
error "This is an error"
warn "This is a warning"
info "This is info"
trace "This is trace"

# Run with different levels:
# DEBUG_LEVEL=1 ./script.sh  # Only errors
# DEBUG_LEVEL=2 ./script.sh  # Errors and warnings
# DEBUG_LEVEL=4 ./script.sh  # Everything
```

## 7. Logging Framework

A comprehensive logging framework for production scripts.

### Simple Logging

```bash
#!/bin/bash

# Log to file
LOG_FILE="/var/log/myscript.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Script started"
log "Processing data..."
log "Script completed"
```

### Multi-Level Logging

```bash
#!/bin/bash

# Log levels
readonly LOG_LEVEL_DEBUG=0
readonly LOG_LEVEL_INFO=1
readonly LOG_LEVEL_WARN=2
readonly LOG_LEVEL_ERROR=3
readonly LOG_LEVEL_FATAL=4

# Current log level
LOG_LEVEL=${LOG_LEVEL:-$LOG_LEVEL_INFO}

# Log file
LOG_FILE="${LOG_FILE:-/var/log/myscript.log}"

# Log function
log_message() {
    local level=$1
    local level_num=$2
    shift 2
    local message=$*

    # Only log if level is high enough
    if [ "$level_num" -ge "$LOG_LEVEL" ]; then
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
    fi
}

# Convenience functions
debug() { log_message "DEBUG" $LOG_LEVEL_DEBUG "$*"; }
info()  { log_message "INFO"  $LOG_LEVEL_INFO "$*"; }
warn()  { log_message "WARN"  $LOG_LEVEL_WARN "$*"; }
error() { log_message "ERROR" $LOG_LEVEL_ERROR "$*"; }
fatal() { log_message "FATAL" $LOG_LEVEL_FATAL "$*"; exit 1; }

# Usage
debug "Debug message"
info "Info message"
warn "Warning message"
error "Error message"
# fatal "Fatal error"  # Uncomment to test
```

### Logging to File and Console

```bash
#!/bin/bash

LOG_FILE="/var/log/myscript.log"

# Setup logging
setup_logging() {
    # Create log file if it doesn't exist
    touch "$LOG_FILE" 2>/dev/null || {
        echo "Cannot create log file: $LOG_FILE" >&2
        LOG_FILE="/tmp/myscript.log"
        echo "Using temporary log file: $LOG_FILE" >&2
    }

    # Redirect stdout and stderr to log file AND console
    exec > >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

setup_logging

echo "This goes to both console and log file"
echo "Error message" >&2
```

### Log Rotation

```bash
#!/bin/bash

LOG_FILE="/var/log/myscript.log"
MAX_LOG_SIZE=$((10 * 1024 * 1024))  # 10 MB
MAX_LOG_FILES=5

rotate_logs() {
    local log_file=$1
    local max_size=$2
    local max_files=$3

    # Check if rotation needed
    if [ ! -f "$log_file" ]; then
        return 0
    fi

    local size=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null)
    if [ "$size" -lt "$max_size" ]; then
        return 0
    fi

    # Rotate old logs
    local i=$max_files
    while [ $i -gt 1 ]; do
        local prev=$((i - 1))
        if [ -f "${log_file}.${prev}" ]; then
            mv "${log_file}.${prev}" "${log_file}.${i}"
        fi
        ((i--))
    done

    # Move current log
    mv "$log_file" "${log_file}.1"
    touch "$log_file"

    echo "Log rotated at $(date)" >> "$log_file"
}

# Check and rotate logs before starting
rotate_logs "$LOG_FILE" "$MAX_LOG_SIZE" "$MAX_LOG_FILES"

# Now log normally
echo "Log entry at $(date)" >> "$LOG_FILE"
```

### Structured Logging

```bash
#!/bin/bash

# Structured logging with key=value pairs
structured_log() {
    local level=$1
    shift

    local timestamp=$(date -u '+%Y-%m-%dT%H:%M:%S.%3NZ')
    local pid=$$
    local script=$(basename "$0")

    # Start with standard fields
    local log_entry="timestamp=$timestamp level=$level pid=$pid script=$script"

    # Add custom fields
    while [ $# -gt 0 ]; do
        log_entry="$log_entry $1"
        shift
    done

    echo "$log_entry"
}

# Usage
structured_log INFO "event=startup" "version=1.0.0"
structured_log INFO "event=processing" "user=john" "action=login" "status=success"
structured_log ERROR "event=error" "error=connection_failed" "host=db.example.com"

# Output:
# timestamp=2024-01-15T10:30:45.123Z level=INFO pid=12345 script=myscript.sh event=startup version=1.0.0
# timestamp=2024-01-15T10:30:46.234Z level=INFO pid=12345 script=myscript.sh event=processing user=john action=login status=success
# timestamp=2024-01-15T10:30:47.345Z level=ERROR pid=12345 script=myscript.sh event=error error=connection_failed host=db.example.com

# This format is easily parseable by log analysis tools
```

## 8. Practice Problems

### Problem 1: Robust File Processor

Write a script that processes multiple files with comprehensive error handling. The script should:
- Accept a directory path as argument
- Validate the directory exists and is readable
- Process each `.txt` file in the directory
- Use `set -euo pipefail`
- Implement proper error handling with trap
- Log all operations (success and failure) to a log file
- Clean up temporary files on exit (normal or error)
- Handle Ctrl+C gracefully with cleanup

### Problem 2: Input Validation Library

Create a reusable input validation library with functions to validate:
- Email addresses (RFC-compliant regex)
- Phone numbers (US format: (123) 456-7890)
- URLs (http/https)
- Credit card numbers (Luhn algorithm)
- Dates (YYYY-MM-DD format, valid date)
- Each function should return 0 for valid, 1 for invalid
- Include error messages explaining what's wrong
- Write tests for each validation function

### Problem 3: Database Backup with Error Recovery

Write a backup script that:
- Connects to a PostgreSQL database
- Creates a backup with pg_dump
- Compresses the backup
- Uploads to S3 (or copies to remote server)
- Verifies the backup integrity
- Uses try-catch pattern to handle each step
- Retries failed operations up to 3 times with exponential backoff
- Sends notification (email or log) on success or failure
- Implements proper cleanup of temporary files
- Uses structured logging

### Problem 4: ShellCheck CI Integration

Create a CI script that:
- Finds all `.sh` files in a repository
- Runs shellcheck on each file
- Collects and formats the results
- Fails CI if any errors (not warnings) are found
- Generates a summary report
- Optionally generates HTML report
- Allows excluding specific files/directories
- Supports custom shellcheck configuration

### Problem 5: Debug Mode Framework

Implement a comprehensive debug mode framework that:
- Accepts DEBUG environment variable with levels: 0 (none), 1 (error), 2 (warn), 3 (info), 4 (debug), 5 (trace)
- Uses different colors for each level
- Includes timestamps, line numbers, and function names in trace output
- Logs to both console and file
- Implements log rotation
- Provides functions: debug(), info(), warn(), error(), fatal()
- Supports structured logging with key=value pairs
- Can be enabled/disabled for specific sections of code
- Includes performance timing (duration of operations)

## Exercises

### Exercise 1: Understand set -e Gotchas

Run the following script and predict the output before executing it. Then explain why each `echo` does or does not execute given `set -e` is active.

```bash
#!/bin/bash
set -e

echo "Start"
false || echo "A"
if false; then echo "B"; fi
echo "C"
! false
echo "D"
false | true
echo "E"
```

Write down which letters (A through E) are printed and explain the rule that governs each case.

### Exercise 2: Build a trap ERR Handler

Write a script called `safe_run.sh` that:
- Enables strict mode (`set -euo pipefail`)
- Installs a `trap ERR` handler that prints: the exit code, the failing command (`$BASH_COMMAND`), and the line number (`$LINENO`)
- Also installs a `trap EXIT` handler that always prints "Cleanup done" regardless of success or failure
- Runs three commands: `echo "step 1"`, `false`, `echo "step 3"`

Verify that step 3 never prints, the ERR trap fires with accurate context, and the EXIT trap always fires.

### Exercise 3: Write a die/assert Library

Create a reusable shell library file `error_lib.sh` that provides:
- Named exit code constants: `E_SUCCESS=0`, `E_GENERAL=1`, `E_INVALID_ARGS=2`, `E_NOT_FOUND=66`
- A `die <code> <message>` function that prints the message to stderr and exits with the given code
- An `assert_file_exists <path>` function that calls `die $E_NOT_FOUND` if the file is missing
- An `assert_not_empty <value> <name>` function that calls `die $E_INVALID_ARGS` if the value is empty

Write a second script that sources `error_lib.sh` and calls each function with both valid and invalid inputs to demonstrate the behavior.

### Exercise 4: Implement Safe File Operations with Cleanup

Write a script that processes a source file and produces a modified output file using only safe operations:
- Use `mktemp` to create a temporary file and register it for cleanup with `trap ... EXIT`
- Write processed content to the temp file first, then atomically move it to the final destination with `mv`
- If any step fails, the trap must remove the temp file and print "Aborted, temp files cleaned"
- Test by intentionally failing midway (e.g., by piping through a command that does not exist) to verify cleanup occurs

### Exercise 5: Multi-Level Debug Logging Framework

Build a logging library `log_lib.sh` with these features:
- A `LOG_LEVEL` environment variable controlling verbosity (0=none, 1=error, 2=warn, 3=info, 4=debug)
- Functions `log_error`, `log_warn`, `log_info`, `log_debug` that each check `LOG_LEVEL` before printing
- Each line includes a timestamp (`date '+%Y-%m-%d %H:%M:%S'`), the level name, and the message
- A `log_structured` function that accepts `key=value` pairs and appends them after the standard prefix

Test by running your script with `LOG_LEVEL=1 ./script.sh` (only errors), `LOG_LEVEL=3` (up to info), and `LOG_LEVEL=4` (all messages). Confirm only the expected lines appear at each level.

---

**Previous**: [Process Management and Job Control](./09_Process_Management.md) | **Next**: [Argument Parsing and CLI Interfaces](./11_Argument_Parsing.md)
