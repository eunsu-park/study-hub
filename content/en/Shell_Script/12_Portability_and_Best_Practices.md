# Lesson 12: Portability and Best Practices

**Difficulty**: ⭐⭐⭐⭐

**Previous**: [Argument Parsing and CLI Interfaces](./11_Argument_Parsing.md) | **Next**: [Shell Script Testing](./13_Testing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between POSIX sh, Bash, and Zsh feature sets and select the appropriate shell for a given task
2. Identify common Bashisms (arrays, `[[ ]]`, process substitution, here strings) and rewrite them as POSIX-compatible equivalents
3. Apply the Google Shell Style Guide conventions for naming, quoting, indentation, and function documentation
4. Implement security best practices including input sanitization, eval avoidance, secure temporary files, and PATH hardening
5. Optimize script performance by minimizing external commands, avoiding unnecessary subshells, and using built-in parameter expansion
6. Organize scripts into modular, maintainable structures with constants, helpers, core logic, and main entry points
7. Write dependency checks with version comparison and graceful degradation when optional tools are missing
8. Prepare scripts for distribution with self-extracting archives, man pages, and bash-completion support

---

Scripts that work perfectly on your development machine can break on a colleague's laptop, a CI runner, or a minimal Docker container. Shell portability, coding standards, and security practices are what turn a personal utility into a reliable piece of shared infrastructure. This lesson covers the conventions and techniques that professional shell developers use to write scripts that are portable, secure, performant, and maintainable across diverse environments.

## 1. POSIX sh vs Bash vs Zsh

Understanding differences between shells ensures portability.

### Feature Comparison

| Feature | POSIX sh | Bash | Zsh | Notes |
|---------|----------|------|-----|-------|
| Arrays | No | Yes | Yes | sh: use positional params |
| `[[ ]]` test | No | Yes | Yes | sh: use `[ ]` |
| Process substitution | No | Yes | Yes | `<(command)` |
| Here strings | No | Yes | Yes | `<<< "string"` |
| `local` keyword | No* | Yes | Yes | *Widely supported but not POSIX |
| `function` keyword | No | Yes | Yes | POSIX uses `name() { }` |
| `$RANDOM` | No | Yes | Yes | sh: use `/dev/urandom` |
| `source` | No | Yes | Yes | sh: use `.` |
| `echo -n` | No* | Yes | Yes | *Use `printf` instead |
| Arithmetic `$(( ))` | Yes | Yes | Yes | POSIX-compliant |
| Parameter expansion | Basic | Extended | Extended | Bash has more patterns |
| Associative arrays | No | Yes (4.0+) | Yes | sh: use eval tricks |
| `&>>` redirection | No | Yes | Yes | sh: use `>>file 2>&1` |
| `time` keyword | No | Yes | Yes | sh: use `/usr/bin/time` |
| `select` loop | No | Yes | Yes | sh: manual menu |

### Shell Detection

```bash
#!/bin/sh

# Detect which shell is running this script
detect_shell() {
    if [ -n "$BASH_VERSION" ]; then
        echo "Running in Bash: $BASH_VERSION"
    elif [ -n "$ZSH_VERSION" ]; then
        echo "Running in Zsh: $ZSH_VERSION"
    elif [ -n "$KSH_VERSION" ]; then
        echo "Running in Ksh: $KSH_VERSION"
    else
        echo "Running in unknown shell (possibly POSIX sh)"
    fi
}

detect_shell

# Check if running in Bash
is_bash() {
    [ -n "$BASH_VERSION" ]
}

if is_bash; then
    echo "Bash-specific features available"
else
    echo "Using POSIX-compatible features only"
fi
```

### When to Use Which Shell

```bash
#!/bin/sh

# Use POSIX sh when:
# - Maximum portability required
# - Script must run on embedded systems
# - Minimal dependencies
# - Alpine Linux (uses busybox sh)

# Use Bash when:
# - Advanced features needed (arrays, associative arrays)
# - Better error handling ([[ ]], set -euo pipefail)
# - More readable code
# - Linux systems (Bash is ubiquitous)

# Use Zsh when:
# - Interactive features needed
# - macOS default shell (10.15+)
# - Advanced globbing required

# Shebang choices:
#!/bin/sh          # POSIX sh (maximum portability)
#!/bin/bash        # Bash (common location)
#!/usr/bin/env bash  # Bash (portable location lookup)
```

## 2. Common Bashisms to Avoid

Bashisms are Bash-specific features that don't work in POSIX sh.

### Test Operators: [[ ]] vs [ ]

```bash
#!/bin/sh

# BAD (Bashism): [[ ]]
# [[ $var == "value" ]]

# GOOD (POSIX): [ ]
var="value"
if [ "$var" = "value" ]; then
    echo "Match"
fi

# BAD (Bashism): [[ with pattern matching ]]
# [[ $file == *.txt ]]

# GOOD (POSIX): use case
case "$file" in
    *.txt) echo "Text file" ;;
    *) echo "Other file" ;;
esac

# BAD (Bashism): [[ with regex ]]
# [[ $string =~ ^[0-9]+$ ]]

# GOOD (POSIX): use grep
if echo "$string" | grep -qE '^[0-9]+$'; then
    echo "Number"
fi

# BAD (Bashism): [[ with && ]]
# [[ -f file && -r file ]]

# GOOD (POSIX): separate [ ] or use -a
if [ -f file ] && [ -r file ]; then
    echo "File exists and is readable"
fi

# Alternative (but [ ] is deprecated):
if [ -f file -a -r file ]; then
    echo "File exists and is readable"
fi
```

### Arrays

```bash
#!/bin/sh

# BAD (Bashism): arrays
# array=(one two three)
# echo "${array[1]}"

# GOOD (POSIX): use positional parameters
set -- one two three
echo "$2"  # Prints: two

# GOOD (POSIX): use space-separated string
items="one two three"
for item in $items; do
    echo "$item"
done

# GOOD (POSIX): use newline-separated string
items="one
two
three"

IFS='
'
for item in $items; do
    echo "$item"
done
```

### Process Substitution

```bash
#!/bin/sh

# BAD (Bashism): process substitution
# diff <(sort file1) <(sort file2)

# GOOD (POSIX): use temporary files
tmp1=$(mktemp)
tmp2=$(mktemp)
trap 'rm -f "$tmp1" "$tmp2"' EXIT

sort file1 > "$tmp1"
sort file2 > "$tmp2"
diff "$tmp1" "$tmp2"
```

### Here Strings

```bash
#!/bin/sh

# BAD (Bashism): here string
# grep "pattern" <<< "$variable"

# GOOD (POSIX): echo with pipe
echo "$variable" | grep "pattern"

# GOOD (POSIX): here document
grep "pattern" << EOF
$variable
EOF

# GOOD (POSIX): printf with pipe
printf '%s\n' "$variable" | grep "pattern"
```

### $RANDOM

```bash
#!/bin/sh

# BAD (Bashism): $RANDOM
# random_num=$RANDOM

# GOOD (POSIX): /dev/urandom
random_num=$(od -An -N2 -i /dev/urandom | tr -d ' ')

# GOOD (POSIX): awk with /dev/urandom
random_num=$(awk 'BEGIN{srand(); print int(rand()*32768)}')

# GOOD (POSIX): hexdump
random_num=$(hexdump -n 2 -e '/2 "%u"' /dev/urandom)
```

### source vs .

```bash
#!/bin/sh

# BAD (Bashism): source
# source ./config.sh

# GOOD (POSIX): .
. ./config.sh

# Both work in Bash, but only . is POSIX
```

### echo vs printf

```bash
#!/bin/sh

# BAD (not portable): echo -n
# echo -n "No newline"

# GOOD (POSIX): printf
printf "No newline"

# BAD (not portable): echo with backslashes
# echo "Line 1\nLine 2"

# GOOD (POSIX): printf
printf "Line 1\nLine 2\n"

# echo is only safe for simple strings without flags or escapes
echo "Simple string"  # OK

# printf is always safe and portable
printf '%s\n' "Any string"  # Always works
```

### function Keyword

```bash
#!/bin/sh

# BAD (Bashism): function keyword
# function my_func() {
#     echo "Hello"
# }

# GOOD (POSIX): no function keyword
my_func() {
    echo "Hello"
}

# Call it
my_func
```

### local Variables

```bash
#!/bin/sh

# BAD (not POSIX, but widely supported): local
# my_func() {
#     local var="value"
# }

# GOOD (POSIX): no local keyword (variables are global)
my_func() {
    # Use prefixed names to avoid conflicts
    _myfunc_var="value"
    echo "$_myfunc_var"
}

# Alternative: use subshell for isolation
my_func_isolated() {
    (
        var="value"  # Only exists in subshell
        echo "$var"
    )
}

# Note: local is so widely supported that it's often used anyway
# even in "POSIX" scripts. Busybox sh supports it, for example.
```

### Complete POSIX vs Bash Example

```bash
#!/bin/sh
# POSIX-compatible version

# Check if file is readable text file
is_readable_text_file() {
    file="$1"

    # Check exists and is regular file
    if [ ! -f "$file" ]; then
        return 1
    fi

    # Check readable
    if [ ! -r "$file" ]; then
        return 1
    fi

    # Check if text file (using file command)
    case "$(file -b "$file")" in
        *text*) return 0 ;;
        *) return 1 ;;
    esac
}

# Bash version (simpler)
#!/bin/bash
is_readable_text_file() {
    local file=$1
    [[ -f "$file" && -r "$file" && $(file -b "$file") == *text* ]]
}
```

## 3. Google Shell Style Guide Highlights

Google's Shell Style Guide provides industry best practices.

### File Header

```bash
#!/bin/bash
#
# Script name: deploy.sh
# Description: Deploys application to production
# Author: John Doe <john@example.com>
# Date: 2024-01-15
# Version: 1.0.0
#
# Usage: deploy.sh [--dry-run] [--environment ENV] VERSION
#
# Copyright 2024 Company Name
# License: MIT

set -euo pipefail
```

### Function Comments

```bash
#!/bin/bash

#######################################
# Processes a file and generates output.
# Globals:
#   OUTPUT_DIR
# Arguments:
#   $1 - Input file path
#   $2 - Output format (json|xml)
# Outputs:
#   Writes processed data to OUTPUT_DIR
# Returns:
#   0 on success, 1 on error
#######################################
process_file() {
    local input_file=$1
    local format=$2

    # Implementation...
}

#######################################
# Cleanup function for trap.
# Globals:
#   TEMP_FILES
# Arguments:
#   None
# Outputs:
#   Cleanup messages to stderr
#######################################
cleanup() {
    # Implementation...
}
```

### TODO Comments

```bash
#!/bin/bash

# TODO(username): Add error handling for network failures
# TODO(username): Implement retry logic with exponential backoff

# FIXME(username): This breaks when file has spaces in name
process_file() {
    # ...
}

# NOTE: This function is deprecated, use process_file_v2 instead
process_file_v1() {
    # ...
}

# HACK: Temporary workaround for bug in external tool
# Will be removed when tool is updated
workaround() {
    # ...
}
```

### Naming Conventions

```bash
#!/bin/bash

# Constants: UPPER_CASE
readonly MAX_RETRIES=3
readonly DEFAULT_TIMEOUT=30
readonly CONFIG_FILE="/etc/app/config.conf"

# Environment variables: UPPER_CASE
export PATH="/usr/local/bin:$PATH"
export DEBUG_MODE=0

# Variables: lowercase_with_underscores
user_name="john"
file_path="/tmp/file.txt"
retry_count=0

# Functions: lowercase_with_underscores
process_file() {
    local input_file=$1
    # ...
}

calculate_checksum() {
    local file=$1
    # ...
}

# Private functions: _prefix (convention, not enforced)
_internal_helper() {
    # ...
}
```

### Indentation and Formatting

```bash
#!/bin/bash

# Use 2 spaces for indentation
if [ "$condition" = "true" ]; then
  echo "Indented with 2 spaces"
  if [ "$nested" = "true" ]; then
    echo "Nested also 2 spaces"
  fi
fi

# Line length: max 80 characters (flexible to 100)
very_long_command --option1 value1 \
  --option2 value2 \
  --option3 value3

# Pipe formatting
cat file.txt \
  | grep "pattern" \
  | sort \
  | uniq

# Or:
cat file.txt |
  grep "pattern" |
  sort |
  uniq
```

### Quoting Rules

```bash
#!/bin/bash

# Always quote variables
filename="my file.txt"
cat "$filename"  # GOOD
# cat $filename  # BAD - word splitting

# Quote command substitutions
result="$(command)"  # GOOD
# result=$(command)  # BAD - not wrong, but inconsistent

# Single quotes for literal strings
echo 'No expansion happens here: $var'

# Double quotes for strings with variables
echo "Value: $var"

# Quote array expansions
files=("file1.txt" "file2.txt")
process "${files[@]}"  # GOOD
# process ${files[@]}  # BAD

# Don't quote arithmetic expansions
count=$((count + 1))  # GOOD

# Don't quote comparison integers
if [ $count -gt 10 ]; then  # OK (but quoting doesn't hurt)
    echo "Greater than 10"
fi
```

### When to Use Shell vs Python/Perl

```bash
#!/bin/bash

# Use shell for:
# - Simple scripts (< 100 lines)
# - Primarily calling other programs
# - File system operations
# - Simple data processing
# - System administration tasks

# Use Python/Perl for:
# - Complex data structures
# - Text processing with complex logic
# - Scripts > 100 lines
# - Need for libraries (HTTP, JSON, etc.)
# - Complex algorithms
# - Better error handling required
# - Testing required

# Example: When shell is appropriate
#!/bin/bash
# Simple backup script
tar czf "backup_$(date +%Y%m%d).tar.gz" /important/data
aws s3 cp "backup_$(date +%Y%m%d).tar.gz" s3://backups/

# Example: When Python is better
#!/usr/bin/env python3
# Complex data processing with error handling
import json
import requests
import logging

# ... complex logic ...
```

## 4. Security Best Practices

Security considerations for shell scripts.

### Input Sanitization

```bash
#!/bin/bash

# Always validate and sanitize input
sanitize_filename() {
    local filename=$1

    # Remove path components
    filename=$(basename "$filename")

    # Remove dangerous characters
    filename=$(echo "$filename" | tr -cd '[:alnum:]._-')

    # Limit length
    filename=${filename:0:255}

    echo "$filename"
}

# Usage
user_input="../../etc/passwd"
safe_name=$(sanitize_filename "$user_input")
echo "Safe: $safe_name"  # Prints: etcpasswd

# Validate numeric input
validate_number() {
    local input=$1

    if ! [[ "$input" =~ ^[0-9]+$ ]]; then
        echo "Error: Not a valid number" >&2
        return 1
    fi

    echo "$input"
}

# Validate against whitelist
validate_enum() {
    local input=$1
    shift
    local valid_values=("$@")

    for value in "${valid_values[@]}"; do
        if [ "$input" = "$value" ]; then
            return 0
        fi
    done

    echo "Error: Invalid value: $input" >&2
    echo "Valid values: ${valid_values[*]}" >&2
    return 1
}

# Usage
if validate_enum "$user_choice" "start" "stop" "restart"; then
    echo "Valid choice"
fi
```

### Quoting to Prevent Injection

```bash
#!/bin/bash

# BAD: Command injection vulnerability
user_input="file.txt; rm -rf /"
cat $user_input  # DANGEROUS!

# GOOD: Proper quoting
cat "$user_input"  # Safe - treated as single filename

# BAD: SQL injection (if using sqlite3 directly)
query="SELECT * FROM users WHERE name = '$user_input'"
sqlite3 db.sqlite "$query"  # VULNERABLE

# GOOD: Use parameterized queries (via here-doc or file)
sqlite3 db.sqlite << EOF
SELECT * FROM users WHERE name = '$user_input';
EOF

# Better: Use proper tools with parameter binding
# (For complex DB work, use Python/Perl/etc.)

# BAD: HTML injection
echo "<div>$user_input</div>" > output.html

# GOOD: Escape HTML entities
escape_html() {
    local input=$1
    input=${input//&/&amp;}
    input=${input//</&lt;}
    input=${input//>/&gt;}
    input=${input//\"/&quot;}
    input=${input//\'/&#39;}
    echo "$input"
}

safe_input=$(escape_html "$user_input")
echo "<div>$safe_input</div>" > output.html
```

### Avoiding eval

```bash
#!/bin/bash

# BAD: eval is dangerous
user_command="rm -rf /"
eval "$user_command"  # NEVER DO THIS

# GOOD: Use case or if/else instead
case "$user_command" in
    start) start_service ;;
    stop) stop_service ;;
    *) echo "Unknown command" >&2 ;;
esac

# BAD: Dynamic variable names with eval
eval "${prefix}_var=value"

# GOOD: Use associative arrays (Bash 4+)
declare -A config
config["${prefix}_var"]="value"

# If you MUST use eval, validate thoroughly
safe_eval() {
    local cmd=$1

    # Whitelist of allowed commands
    case "$cmd" in
        "echo "*)
            eval "$cmd"
            ;;
        *)
            echo "Command not allowed" >&2
            return 1
            ;;
    esac
}
```

### Secure Temporary Files

```bash
#!/bin/bash

# BAD: Predictable temp file names
tmpfile="/tmp/myapp_$$"  # $$ is predictable

# GOOD: Use mktemp
tmpfile=$(mktemp) || exit 1
trap 'rm -f "$tmpfile"' EXIT

echo "data" > "$tmpfile"

# GOOD: Temp directory
tmpdir=$(mktemp -d) || exit 1
trap 'rm -rf "$tmpdir"' EXIT

# Set restrictive permissions
chmod 700 "$tmpdir"

# Create files in temp directory
echo "data" > "$tmpdir/file1.txt"
echo "more" > "$tmpdir/file2.txt"
```

### PATH Hardening

```bash
#!/bin/bash

# Set secure PATH
export PATH="/usr/local/bin:/usr/bin:/bin"

# Or use absolute paths for critical commands
/usr/bin/whoami
/bin/cat /etc/passwd

# Verify command location
require_command() {
    local cmd=$1
    local path

    path=$(command -v "$cmd")
    if [ -z "$path" ]; then
        echo "Error: Command not found: $cmd" >&2
        exit 1
    fi

    # Verify it's in expected location
    case "$path" in
        /usr/bin/*|/bin/*|/usr/local/bin/*)
            echo "Using: $path" >&2
            ;;
        *)
            echo "Warning: Command in unexpected location: $path" >&2
            ;;
    esac
}

require_command "grep"
require_command "awk"
```

### Running as Least Privilege

```bash
#!/bin/bash

# Check if running as root when not needed
if [ "$(id -u)" -eq 0 ]; then
    echo "Error: Do not run this script as root" >&2
    exit 1
fi

# Drop privileges if started as root
drop_privileges() {
    local user=$1

    if [ "$(id -u)" -eq 0 ]; then
        echo "Dropping privileges to user: $user"
        exec su - "$user" -c "$0 $*"
    fi
}

# Require root for certain operations
require_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "Error: This script must be run as root" >&2
        exit 1
    fi
}

# Check for sudo
has_sudo() {
    sudo -n true 2>/dev/null
}

# Use sudo for specific commands only
safe_system_update() {
    if has_sudo; then
        sudo apt-get update
        sudo apt-get upgrade -y
    else
        echo "Error: sudo access required" >&2
        return 1
    fi
}
```

## 5. Performance Optimization

Writing efficient shell scripts.

### Minimize External Commands

```bash
#!/bin/bash

# BAD: External command for string manipulation
basename=$(basename "$path")
dirname=$(dirname "$path")

# GOOD: Built-in parameter expansion
basename=${path##*/}
dirname=${path%/*}

# BAD: Using grep/sed for simple checks
if echo "$string" | grep -q "pattern"; then
    echo "Found"
fi

# GOOD: Built-in pattern matching (Bash)
if [[ "$string" == *pattern* ]]; then
    echo "Found"
fi

# BAD: Multiple echo calls
echo "Line 1"
echo "Line 2"
echo "Line 3"

# GOOD: Single printf or here-doc
printf '%s\n' "Line 1" "Line 2" "Line 3"

# Or:
cat << EOF
Line 1
Line 2
Line 3
EOF
```

### Avoid Subshells

```bash
#!/bin/bash

# BAD: Subshell in loop
count=0
cat file.txt | while read line; do
    ((count++))
done
echo "$count"  # Prints 0 (subshell!)

# GOOD: Process substitution or redirection
count=0
while read line; do
    ((count++))
done < file.txt
echo "$count"  # Correct count

# BAD: Command substitution for variable assignment in loop
for i in {1..1000}; do
    date_str=$(date +%Y%m%d)  # Called 1000 times!
done

# GOOD: Calculate once
date_str=$(date +%Y%m%d)
for i in {1..1000}; do
    # Use $date_str
done
```

### Use mapfile/readarray

```bash
#!/bin/bash

# BAD: Reading file in loop with command substitution
files=()
while IFS= read -r line; do
    files+=("$line")
done < <(find . -name "*.txt")

# GOOD: Use mapfile (Bash 4+)
mapfile -t files < <(find . -name "*.txt")

# Or readarray (same as mapfile)
readarray -t files < <(find . -name "*.txt")

# Read from file
mapfile -t lines < file.txt

# Skip empty lines
mapfile -t lines < <(grep -v '^$' file.txt)
```

### Batch Operations

```bash
#!/bin/bash

# BAD: Multiple operations in loop
for file in *.txt; do
    chmod 644 "$file"
done

# GOOD: Batch operation
chmod 644 *.txt

# BAD: Individual git operations
for commit in $commits; do
    git show "$commit"
done

# GOOD: Single git command
git show $commits

# BAD: Multiple curl requests in sequence
for url in $urls; do
    curl "$url"
done

# GOOD: Parallel curl with xargs
echo "$urls" | xargs -P 4 -n 1 curl
```

### Benchmarking

```bash
#!/bin/bash

# Simple timing
echo "Method 1:"
time {
    for i in {1..1000}; do
        basename=$(basename "/path/to/file$i.txt")
    done
}

echo "Method 2:"
time {
    for i in {1..1000}; do
        basename="${path##*/}"
    done
}

# More detailed timing
benchmark() {
    local iterations=${1:-100}
    local code=$2

    local start=$(date +%s%N)

    for ((i=0; i<iterations; i++)); do
        eval "$code"
    done

    local end=$(date +%s%N)
    local duration=$(((end - start) / 1000000))  # Convert to milliseconds

    echo "Duration: ${duration}ms for $iterations iterations"
    echo "Average: $((duration / iterations))ms per iteration"
}

# Usage
benchmark 1000 'x=$(basename "/path/to/file.txt")'
benchmark 1000 'x=${path##*/}'
```

## 6. Code Organization

Structuring scripts for maintainability.

### Script Template

```bash
#!/bin/bash
#
# Script: script_name.sh
# Description: Brief description
# Author: Your Name
# Date: 2024-01-15
#

set -euo pipefail

# ============================================================================
# CONSTANTS
# ============================================================================

readonly SCRIPT_NAME=$(basename "$0")
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly VERSION="1.0.0"

readonly DEFAULT_TIMEOUT=30
readonly MAX_RETRIES=3

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

VERBOSE=0
DRY_RUN=0
CONFIG_FILE=""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

error() {
    log "ERROR: $*"
}

die() {
    error "$*"
    exit 1
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

main_function_1() {
    # Implementation
    :
}

main_function_2() {
    # Implementation
    :
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    # Parse arguments
    # Validate input
    # Execute main logic
    :
}

# Only run main if script is executed, not sourced
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
```

### Modular Design

```bash
#!/bin/bash

# config.sh - Configuration module
load_config() {
    local config_file=$1

    if [ ! -f "$config_file" ]; then
        echo "Config file not found: $config_file" >&2
        return 1
    fi

    # Source config file
    # shellcheck source=/dev/null
    . "$config_file"
}

# logger.sh - Logging module
setup_logger() {
    LOG_FILE=${LOG_FILE:-/var/log/app.log}
    LOG_LEVEL=${LOG_LEVEL:-INFO}
}

log_message() {
    local level=$1
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" >> "$LOG_FILE"
}

# network.sh - Network module
check_connectivity() {
    local host=$1
    ping -c 1 -W 2 "$host" &>/dev/null
}

download_file() {
    local url=$1
    local output=$2
    curl -fsSL -o "$output" "$url"
}

# main.sh - Main script
#!/bin/bash

# Source modules
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/config.sh"
. "$SCRIPT_DIR/logger.sh"
. "$SCRIPT_DIR/network.sh"

main() {
    load_config "/etc/app/config.conf"
    setup_logger

    if check_connectivity "example.com"; then
        log_message INFO "Network available"
    fi
}

main "$@"
```

### Configuration Management

```bash
#!/bin/bash

# Load configuration from file
load_config_ini() {
    local config_file=$1

    if [ ! -f "$config_file" ]; then
        return 1
    fi

    # Parse INI-style config
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue

        # Remove leading/trailing whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)

        # Export as environment variable
        export "$key=$value"
    done < "$config_file"
}

# Generate config file
generate_config() {
    local output_file=$1

    cat > "$output_file" << 'EOF'
# Application Configuration

# Database settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=myapp
DB_USER=appuser

# API settings
API_KEY=your_api_key_here
API_TIMEOUT=30

# Application settings
DEBUG_MODE=0
LOG_LEVEL=INFO
MAX_CONNECTIONS=100
EOF

    echo "Configuration written to: $output_file"
}

# Load config from environment, file, or defaults
load_config_priority() {
    # 1. Defaults
    DB_HOST=${DB_HOST:-localhost}
    DB_PORT=${DB_PORT:-5432}
    API_TIMEOUT=${API_TIMEOUT:-30}

    # 2. Config file (if exists)
    if [ -f "/etc/app/config.conf" ]; then
        # shellcheck source=/dev/null
        . "/etc/app/config.conf"
    fi

    # 3. Environment variables (highest priority, already loaded)

    # Export final values
    export DB_HOST DB_PORT API_TIMEOUT
}
```

## 7. Dependency Management

Managing script dependencies.

### Checking Required Commands

```bash
#!/bin/bash

# Check single command
require_command() {
    local cmd=$1
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: Required command not found: $cmd" >&2
        return 1
    fi
}

# Check multiple commands
require_commands() {
    local missing=()

    for cmd in "$@"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo "Error: Required commands not found:" >&2
        printf '  - %s\n' "${missing[@]}" >&2
        return 1
    fi
}

# Check with installation instructions
require_command_with_hint() {
    local cmd=$1
    local install_cmd=$2

    if ! command -v "$cmd" &>/dev/null; then
        cat << EOF >&2
Error: Required command not found: $cmd

To install, run:
    $install_cmd

EOF
        return 1
    fi
}

# Usage
require_commands grep sed awk curl jq || exit 1

require_command_with_hint "jq" "apt-get install jq" || exit 1
require_command_with_hint "docker" "curl -fsSL https://get.docker.com | sh" || exit 1
```

### Version Checking

```bash
#!/bin/bash

# Get command version
get_version() {
    local cmd=$1

    case "$cmd" in
        bash)
            bash --version | head -1 | grep -oP '\d+\.\d+\.\d+'
            ;;
        python*)
            $cmd --version 2>&1 | grep -oP '\d+\.\d+\.\d+'
            ;;
        git)
            git --version | grep -oP '\d+\.\d+\.\d+'
            ;;
        docker)
            docker --version | grep -oP '\d+\.\d+\.\d+'
            ;;
        *)
            echo "Unknown command: $cmd" >&2
            return 1
            ;;
    esac
}

# Compare versions
version_compare() {
    local version1=$1
    local version2=$2

    if [ "$version1" = "$version2" ]; then
        echo "0"
        return
    fi

    local IFS=.
    local i ver1=($version1) ver2=($version2)

    # Fill empty positions with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done

    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            ver2[i]=0
        fi

        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            echo "1"
            return
        fi

        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            echo "-1"
            return
        fi
    done

    echo "0"
}

# Require minimum version
require_version() {
    local cmd=$1
    local min_version=$2

    local current_version
    current_version=$(get_version "$cmd") || return 1

    local cmp
    cmp=$(version_compare "$current_version" "$min_version")

    if [ "$cmp" -lt 0 ]; then
        echo "Error: $cmd version $current_version is too old" >&2
        echo "       Minimum required version: $min_version" >&2
        return 1
    fi

    echo "Using $cmd version $current_version" >&2
}

# Usage
require_version "bash" "4.0.0" || exit 1
require_version "git" "2.0.0" || exit 1
```

### Graceful Degradation

```bash
#!/bin/bash

# Use advanced features if available, fallback otherwise
use_color() {
    if [ -t 1 ] && command -v tput &>/dev/null; then
        # Terminal with tput support
        RED=$(tput setaf 1)
        GREEN=$(tput setaf 2)
        RESET=$(tput sgr0)
    elif [ -t 1 ]; then
        # Terminal without tput
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        RESET='\033[0m'
    else
        # No terminal
        RED=''
        GREEN=''
        RESET=''
    fi
}

# Use jq if available, fallback to grep/sed
parse_json() {
    local json_file=$1
    local key=$2

    if command -v jq &>/dev/null; then
        jq -r ".$key" "$json_file"
    else
        # Fallback to grep/sed (fragile but works for simple cases)
        grep "\"$key\"" "$json_file" | sed 's/.*: "\(.*\)".*/\1/'
    fi
}

# Use parallel if available, fallback to xargs
parallel_execute() {
    local cmd=$1
    shift
    local items=("$@")

    if command -v parallel &>/dev/null; then
        printf '%s\n' "${items[@]}" | parallel "$cmd"
    else
        printf '%s\n' "${items[@]}" | xargs -P 4 -I {} sh -c "$cmd {}"
    fi
}
```

## 8. Distribution and Packaging

Preparing scripts for distribution.

### Single-File Script

```bash
#!/bin/bash
#
# Complete standalone script with everything embedded
#

set -euo pipefail

# Embed configuration
read -r -d '' DEFAULT_CONFIG << 'EOF' || true
DB_HOST=localhost
DB_PORT=5432
EOF

# Embed helper functions
error() { echo "ERROR: $*" >&2; }
info() { echo "INFO: $*"; }

# Main logic
main() {
    info "Starting application"
    # ...
}

main "$@"
```

### Self-Extracting Archive

```bash
#!/bin/bash
# Self-extracting script with embedded tarball

ARCHIVE_LINE=$(awk '/^__ARCHIVE__/ {print NR + 1; exit 0; }' "$0")

# Extract embedded archive
tail -n +${ARCHIVE_LINE} "$0" | tar xz -C /tmp

# Run installer
cd /tmp/installer
./install.sh

exit 0

__ARCHIVE__
# Compressed tarball data starts here (created with tar czf)
```

### Create the self-extracting archive:

```bash
#!/bin/bash

# Create installer package
create_installer() {
    local output_file=$1
    local source_dir=$2

    # Create header script
    cat > "$output_file" << 'HEADER'
#!/bin/bash
ARCHIVE_LINE=$(awk '/^__ARCHIVE__/ {print NR + 1; exit 0; }' "$0")
tail -n +${ARCHIVE_LINE} "$0" | tar xz -C /tmp
cd /tmp/installer && ./install.sh
exit 0
__ARCHIVE__
HEADER

    # Append tarball
    tar czf - -C "$source_dir" . >> "$output_file"

    chmod +x "$output_file"
    echo "Created: $output_file"
}

create_installer "install.sh" "./installer_files"
```

### Man Page

```bash
#!/bin/bash

# Generate man page
generate_manpage() {
    local cmd_name=$1
    local output_file=$2

    cat > "$output_file" << 'EOF'
.TH MYTOOL 1 "January 2024" "mytool 1.0.0" "User Commands"
.SH NAME
mytool \- Brief description of mytool
.SH SYNOPSIS
.B mytool
[\fIOPTIONS\fR] \fICOMMAND\fR [\fIARGS\fR]
.SH DESCRIPTION
.B mytool
is a tool that does something useful.
Detailed description goes here.
.SH OPTIONS
.TP
.BR \-v ", " \-\-verbose
Enable verbose output
.TP
.BR \-h ", " \-\-help
Show help message
.SH EXAMPLES
.TP
mytool process file.txt
Process a file
.TP
mytool \-v \-\-output=result.txt input.txt
Verbose processing with output file
.SH EXIT STATUS
.TP
.B 0
Success
.TP
.B 1
General error
.SH AUTHOR
Written by Your Name.
.SH REPORTING BUGS
Report bugs to: bugs@example.com
.SH SEE ALSO
Full documentation at: https://example.com/docs
EOF

    echo "Man page created: $output_file"
    echo "Install with: cp $output_file /usr/local/share/man/man1/"
    echo "View with: man $cmd_name"
}

generate_manpage "mytool" "mytool.1"
```

### Bash Completion

```bash
#!/bin/bash

# Bash completion script for mytool
_mytool_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Options
    opts="-v --verbose -o --output -h --help --version"

    # Commands
    commands="start stop restart status"

    # Complete options
    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi

    # Complete commands
    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
        return 0
    fi

    # Complete files for --output
    if [[ ${prev} == "-o" ]] || [[ ${prev} == "--output" ]]; then
        COMPREPLY=( $(compgen -f -- ${cur}) )
        return 0
    fi

    # Default: complete files
    COMPREPLY=( $(compgen -f -- ${cur}) )
}

# Register completion
complete -F _mytool_completion mytool

# Install instructions:
# cp mytool-completion.bash /etc/bash_completion.d/mytool
```

## 9. Practice Problems

### Problem 1: POSIX Shell Converter

Create a tool that:
- Analyzes a Bash script and identifies Bashisms
- Suggests POSIX-compatible alternatives
- Optionally attempts automatic conversion
- Generates a report of changes needed
- Tests the converted script for syntax errors
- Maintains functionality through testing

### Problem 2: Performance Profiler

Build a profiling tool that:
- Instruments a shell script to measure execution time of each function
- Identifies bottlenecks (slowest functions)
- Counts how many times each command is called
- Suggests optimizations based on analysis
- Generates a visual report (HTML or terminal-based)
- Compares "before" and "after" performance

### Problem 3: Security Auditor

Develop a security audit tool that:
- Scans scripts for common vulnerabilities (eval, unquoted variables, etc.)
- Checks for insecure temp file creation
- Identifies potential command injection points
- Verifies input validation practices
- Checks file permissions and ownership
- Generates a security report with severity levels
- Suggests fixes for each issue found

### Problem 4: Package Manager

Create a simple package manager for shell scripts that:
- Installs scripts to appropriate directories (`/usr/local/bin`)
- Manages dependencies (checks for required commands)
- Handles version updates
- Generates and installs man pages
- Sets up bash completion
- Supports uninstallation with cleanup
- Maintains a registry of installed scripts

### Problem 5: Test Framework

Implement a testing framework for shell scripts that:
- Supports unit tests for functions
- Mocks external commands
- Captures and validates output (stdout/stderr)
- Tests exit codes
- Provides assertions (assert_equals, assert_contains, etc.)
- Generates coverage reports
- Integrates with CI/CD systems
- Produces JUnit-style XML reports

## Exercises

### Exercise 1: Audit a Script for Portability Issues

Take the following script fragment and identify every portability issue. For each issue, state the problem and provide the portable fix.

```bash
#!/bin/bash
which python3 > /dev/null
result=`python3 -c "print(2**10)"`
echo "Result: $result"
ls *.log | while read file; do
    wc -l $file
done
stat -c%s report.txt
function cleanup { rm -f /tmp/myapp.$$; }
```

Issues to look for: bashisms, unsafe quoting, unportable flags, deprecated syntax, unsafe patterns with special filenames.

### Exercise 2: Write a POSIX-Compatible Script

Rewrite the following bash-specific script to be POSIX-compatible (`#!/bin/sh`). Replace every bash-only feature with a POSIX equivalent.

```bash
#!/bin/bash
declare -a files=()
for f in *.conf; do
    [[ -f "$f" ]] && files+=("$f")
done

process() {
    local name="$1"
    echo "Processing: $name"
    [[ "$name" =~ ^[0-9]+_ ]] && echo "  (numbered file)"
}

for f in "${files[@]}"; do
    process "$f"
done
```

After rewriting, test that it runs with both `bash` and `sh`.

### Exercise 3: Integrate ShellCheck into a Workflow

Set up ShellCheck for a small project:
1. Install ShellCheck (if not already available)
2. Create a `.shellcheckrc` file that sets `shell=bash` and disables `SC2034` (unused variables — common in sourced libraries)
3. Write a `lint.sh` script that finds all `*.sh` files under the current directory, runs `shellcheck` on each, and exits with code 1 if any errors (severity `error`) are found
4. Intentionally introduce two ShellCheck warnings into a test script and verify that `lint.sh` catches them

### Exercise 4: Apply Performance Best Practices

Profile and optimize the following slow script. Measure execution time before and after each change using `time`.

```bash
#!/bin/bash
count=0
while read line; do
    if echo "$line" | grep -q "ERROR"; then
        count=$((count + 1))
    fi
done < application.log
echo "Error count: $count"
```

Apply at least three optimizations (hint: avoid subshells in loops, use built-in string operations, prefer `grep -c` for counting). Compare the final `time` output with the original.

### Exercise 5: Create a Distributable Script Package

Package a script for distribution following best practices:
- Write a `install.sh` that copies `myscript.sh` to `/usr/local/bin`, sets permissions to `755`, and generates a man page entry at `/usr/local/share/man/man1/myscript.1`
- Add a `--prefix` option to `install.sh` to support non-root installation (e.g., `~/.local`)
- Write an `uninstall.sh` that reverses the installation cleanly
- Add a version check: if the system's bash is older than 4.0, print a warning and exit

---

**Previous**: [Argument Parsing and CLI Interfaces](./11_Argument_Parsing.md) | **Next**: [Shell Script Testing](./13_Testing.md)
