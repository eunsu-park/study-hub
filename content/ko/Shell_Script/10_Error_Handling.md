# Lesson 10: 에러 처리 및 디버깅

**난이도**: ⭐⭐⭐

**이전**: [09_Process_Management.md](./09_Process_Management.md) | **다음**: [11_Argument_Parsing.md](./11_Argument_Parsing.md)

---

## 1. set 옵션 심화

`set` 명령어는 셸 동작 및 에러 처리를 제어합니다. 이러한 옵션을 이해하는 것은 견고한 스크립트를 작성하는 데 매우 중요합니다.

### set -e (errexit)

```bash
#!/bin/bash

# Exit immediately if any command returns non-zero
set -e

echo "Starting..."
false  # This will cause the script to exit
echo "This won't be printed"
```

### set -e의 주의사항

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

### set 옵션 비교

| 옵션 | 설명 | 효과 | 사용 시기 |
|--------|-------------|--------|-------------|
| `set -e` | errexit | 명령어 실패 시 종료 | 프로덕션 스크립트 |
| `set -u` | nounset | 정의되지 않은 변수 접근 시 종료 | 오타를 조기에 발견 |
| `set -o pipefail` | pipefail | 파이프라인 내 어떤 명령어라도 실패하면 실패 | `set -e`와 함께 |
| `set -x` | xtrace | 실행 전 명령어 출력 | 디버깅 |
| `set -v` | verbose | 셸 입력 라인 출력 | 심층 디버깅 |
| `set -n` | noexec | 명령어를 읽되 실행하지 않음 | 문법 검사 |
| `set -C` | noclobber | 출력 리디렉션이 파일을 덮어쓰는 것을 방지 | 파일 보호 |

### 권장 스크립트 헤더

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

### set -e 일시적으로 비활성화

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

`ERR` 트랩은 명령어가 0이 아닌 종료 상태를 반환할 때 발생합니다 (`set -e`가 종료하지 않는 경우는 제외).

### 기본 ERR 트랩

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

### 에러 컨텍스트 가져오기

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

### 스택 트레이스 생성

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

### 고급 에러 핸들러

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

### ERR vs EXIT 트랩

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

## 3. 커스텀 에러 프레임워크

재사용 가능한 에러 처리 프레임워크를 구축하면 스크립트의 유지보수성이 향상됩니다.

### 에러 코드 열거형(Enum)

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

### 에러 함수

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

### Try-Catch 시뮬레이션

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

### 완전한 에러 프레임워크

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

## 4. 방어적 코딩 패턴

방어적 코딩은 에러가 발생하기 전에 예방합니다.

### 입력 검증

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

### 명령어 존재 여부 확인

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

### 안전한 임시 파일

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

### 락 파일

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

### 안전한 파일 작업

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

## 5. ShellCheck 정적 분석

ShellCheck은 셸 스크립트에서 버그를 찾아내는 정적 분석 도구입니다.

### 일반적인 ShellCheck 경고

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

### ShellCheck 통합

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

### ShellCheck 설정

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

### 코드 내 경고 억제

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

## 6. 디버깅 기법

효과적인 디버깅 기법은 문제를 신속하게 식별하고 수정하는 데 도움이 됩니다.

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

### 향상된 추적을 위한 커스텀 PS4

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

### 선택적 디버깅

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

### Bash 디버거 (bashdb)

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

### 디버그 로깅

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

## 7. 로깅 프레임워크

프로덕션 스크립트를 위한 포괄적인 로깅 프레임워크입니다.

### 간단한 로깅

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

### 다단계 로깅

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

### 파일 및 콘솔 로깅

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

### 로그 회전

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

### 구조화된 로깅

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

## 8. 연습 문제

### 문제 1: 견고한 파일 프로세서

포괄적인 에러 처리를 갖춘 여러 파일을 처리하는 스크립트를 작성하세요. 스크립트는 다음을 수행해야 합니다:
- 디렉토리 경로를 인수로 받음
- 디렉토리가 존재하고 읽기 가능한지 검증
- 디렉토리 내 각 `.txt` 파일 처리
- `set -euo pipefail` 사용
- trap을 사용한 적절한 에러 처리 구현
- 모든 작업(성공 및 실패)을 로그 파일에 기록
- 종료 시(정상 또는 에러) 임시 파일 정리
- Ctrl+C를 정리와 함께 우아하게 처리

### 문제 2: 입력 검증 라이브러리

다음을 검증하는 함수를 가진 재사용 가능한 입력 검증 라이브러리를 생성하세요:
- 이메일 주소(RFC 호환 정규식)
- 전화번호(미국 형식: (123) 456-7890)
- URL(http/https)
- 신용카드 번호(Luhn 알고리즘)
- 날짜(YYYY-MM-DD 형식, 유효한 날짜)
- 각 함수는 유효하면 0, 무효하면 1 반환
- 무엇이 잘못되었는지 설명하는 에러 메시지 포함
- 각 검증 함수에 대한 테스트 작성

### 문제 3: 에러 복구를 갖춘 데이터베이스 백업

다음을 수행하는 백업 스크립트를 작성하세요:
- PostgreSQL 데이터베이스에 연결
- pg_dump로 백업 생성
- 백업 압축
- S3에 업로드(또는 원격 서버에 복사)
- 백업 무결성 검증
- 각 단계를 처리하기 위해 try-catch 패턴 사용
- 지수 백오프로 실패한 작업을 최대 3회 재시도
- 성공 또는 실패 시 알림(이메일 또는 로그) 전송
- 임시 파일의 적절한 정리 구현
- 구조화된 로깅 사용

### 문제 4: ShellCheck CI 통합

다음을 수행하는 CI 스크립트를 생성하세요:
- 리포지토리에서 모든 `.sh` 파일 찾기
- 각 파일에 shellcheck 실행
- 결과 수집 및 포맷
- 에러(경고 아님)가 발견되면 CI 실패
- 요약 리포트 생성
- 선택적으로 HTML 리포트 생성
- 특정 파일/디렉토리 제외 허용
- 커스텀 shellcheck 설정 지원

### 문제 5: 디버그 모드 프레임워크

다음을 수행하는 포괄적인 디버그 모드 프레임워크를 구현하세요:
- DEBUG 환경 변수를 레벨과 함께 받음: 0(없음), 1(에러), 2(경고), 3(정보), 4(디버그), 5(추적)
- 각 레벨에 다른 색상 사용
- 추적 출력에 타임스탬프, 라인 번호, 함수 이름 포함
- 콘솔 및 파일 모두에 로그
- 로그 회전 구현
- debug(), info(), warn(), error(), fatal() 함수 제공
- key=value 쌍으로 구조화된 로깅 지원
- 코드의 특정 섹션에 대해 활성화/비활성화 가능
- 성능 타이밍(작업 지속 시간) 포함

---

**이전**: [09_Process_Management.md](./09_Process_Management.md) | **다음**: [11_Argument_Parsing.md](./11_Argument_Parsing.md)
