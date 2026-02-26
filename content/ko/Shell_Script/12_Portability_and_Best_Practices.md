# 레슨 12: 이식성과 모범 사례

**난이도**: ⭐⭐⭐⭐

**이전**: [인수 파싱 및 CLI 인터페이스](./11_Argument_Parsing.md) | **다음**: [셸 스크립트 테스팅](./13_Testing.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. POSIX sh, Bash, Zsh의 기능 집합을 구분하고 주어진 작업에 적합한 셸을 선택할 수 있습니다
2. 일반적인 Bash 전용 문법(Bashisms)(배열, `[[ ]]`, 프로세스 치환(process substitution), here strings)을 식별하고 POSIX 호환 방식으로 재작성할 수 있습니다
3. 명명 규칙, 따옴표 처리, 들여쓰기, 함수 문서화에 관한 Google Shell Style Guide 관례를 적용할 수 있습니다
4. 입력 sanitization, `eval` 회피, 안전한 임시 파일 처리, PATH 강화(hardening)를 포함한 보안 모범 사례를 구현할 수 있습니다
5. 외부 명령 최소화, 불필요한 서브셸 회피, 내장 매개변수 확장(parameter expansion) 활용을 통해 스크립트 성능을 최적화할 수 있습니다
6. 상수, 헬퍼 함수, 핵심 로직, 메인 진입점으로 구성된 모듈화되고 유지보수 가능한 구조로 스크립트를 정리할 수 있습니다
7. 버전 비교 기능을 갖춘 의존성 검사와 선택적 도구가 없을 때의 우아한 성능 저하(graceful degradation)를 작성할 수 있습니다
8. 자기 압축 해제 아카이브(self-extracting archives), man 페이지, bash-completion 지원으로 배포용 스크립트를 준비할 수 있습니다

---

개발 환경에서 완벽하게 동작하는 스크립트가 동료의 노트북, CI 러너, 혹은 최소화된 Docker 컨테이너에서는 실패할 수 있습니다. 셸 이식성(portability), 코딩 표준, 보안 관행은 개인 유틸리티를 신뢰할 수 있는 공유 인프라로 탈바꿈시키는 요소입니다. 이 레슨은 전문 셸 개발자들이 다양한 환경에서 이식 가능하고, 안전하며, 성능이 뛰어나고, 유지보수 가능한 스크립트를 작성하기 위해 사용하는 관례와 기법을 다룹니다.

## 1. POSIX sh vs Bash vs Zsh

셸 간의 차이점을 이해하면 이식성을 보장할 수 있습니다.

### 기능 비교

| 기능 | POSIX sh | Bash | Zsh | 참고 사항 |
|---------|----------|------|-----|-------|
| 배열(Arrays) | No | Yes | Yes | sh: 위치 매개변수 사용 |
| `[[ ]]` 테스트 | No | Yes | Yes | sh: `[ ]` 사용 |
| 프로세스 치환(Process substitution) | No | Yes | Yes | `<(command)` |
| Here strings | No | Yes | Yes | `<<< "string"` |
| `local` 키워드 | No* | Yes | Yes | *널리 지원되지만 POSIX는 아님 |
| `function` 키워드 | No | Yes | Yes | POSIX는 `name() { }` 사용 |
| `$RANDOM` | No | Yes | Yes | sh: `/dev/urandom` 사용 |
| `source` | No | Yes | Yes | sh: `.` 사용 |
| `echo -n` | No* | Yes | Yes | *대신 `printf` 사용 |
| 산술 `$(( ))` | Yes | Yes | Yes | POSIX 호환 |
| 매개변수 확장(Parameter expansion) | 기본 | 확장 | 확장 | Bash는 더 많은 패턴 제공 |
| 연관 배열(Associative arrays) | No | Yes (4.0+) | Yes | sh: eval 트릭 사용 |
| `&>>` 리다이렉션 | No | Yes | Yes | sh: `>>file 2>&1` 사용 |
| `time` 키워드 | No | Yes | Yes | sh: `/usr/bin/time` 사용 |
| `select` 루프 | No | Yes | Yes | sh: 수동 메뉴 |

### 셸 감지

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

### 어떤 셸을 사용할지

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

## 2. 피해야 할 일반적인 Bashism

Bashism은 POSIX sh에서 작동하지 않는 Bash 전용 기능입니다.

### 테스트 연산자: [[ ]] vs [ ]

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

### 배열

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

### 프로세스 치환

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

### function 키워드

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

### local 변수

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

### 완전한 POSIX vs Bash 예제

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

## 3. Google Shell Style Guide 하이라이트

Google의 Shell Style Guide는 업계 모범 사례를 제공합니다.

### 파일 헤더

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

### 함수 주석

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

### TODO 주석

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

### 네이밍 규칙

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

### 들여쓰기와 포매팅

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

### 인용 규칙

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

### 셸과 Python/Perl 중 무엇을 사용할지

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

## 4. 보안 모범 사례

셸 스크립트에 대한 보안 고려 사항입니다.

### 입력 검증

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

### 인젝션 방지를 위한 인용

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

### eval 피하기

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

### 안전한 임시 파일

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

### PATH 강화

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

### 최소 권한으로 실행

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

## 5. 성능 최적화

효율적인 셸 스크립트 작성하기.

### 외부 명령 최소화

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

### 서브셸 피하기

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

### mapfile/readarray 사용

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

### 배치 작업

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

### 벤치마킹

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

## 6. 코드 구성

유지보수를 위한 스크립트 구조화.

### 스크립트 템플릿

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

### 모듈형 설계

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

### 구성 관리

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

## 7. 의존성 관리

스크립트 의존성 관리하기.

### 필수 명령어 확인

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

### 버전 확인

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

### 우아한 성능 저하(Graceful Degradation)

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

## 8. 배포 및 패키징

스크립트를 배포용으로 준비하기.

### 단일 파일 스크립트

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

### 자체 압축 해제 아카이브

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

### 자체 압축 해제 아카이브 생성:

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

### Man 페이지

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

### Bash 자동 완성

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

## 9. 연습 문제

### 문제 1: POSIX Shell 변환기

다음을 수행하는 도구를 만드세요:
- Bash 스크립트를 분석하고 Bashism을 식별
- POSIX 호환 대안 제안
- 선택적으로 자동 변환 시도
- 필요한 변경 사항 보고서 생성
- 변환된 스크립트의 구문 오류 테스트
- 테스트를 통해 기능 유지

### 문제 2: 성능 프로파일러

다음을 수행하는 프로파일링 도구를 구축하세요:
- 각 함수의 실행 시간을 측정하도록 셸 스크립트 계측
- 병목 지점(가장 느린 함수) 식별
- 각 명령이 호출된 횟수 계산
- 분석 기반 최적화 제안
- 시각적 보고서 생성(HTML 또는 터미널 기반)
- "이전"과 "이후" 성능 비교

### 문제 3: 보안 감사기

다음을 수행하는 보안 감사 도구를 개발하세요:
- 일반적인 취약점(eval, 인용되지 않은 변수 등) 스캔
- 안전하지 않은 임시 파일 생성 확인
- 잠재적인 명령 인젝션 지점 식별
- 입력 검증 방법 확인
- 파일 권한 및 소유권 확인
- 심각도 수준이 포함된 보안 보고서 생성
- 발견된 각 문제에 대한 수정 제안

### 문제 4: 패키지 관리자

다음을 수행하는 셸 스크립트용 간단한 패키지 관리자를 만드세요:
- 적절한 디렉토리(`/usr/local/bin`)에 스크립트 설치
- 의존성 관리(필수 명령어 확인)
- 버전 업데이트 처리
- man 페이지 생성 및 설치
- bash 자동 완성 설정
- 정리와 함께 제거 지원
- 설치된 스크립트 레지스트리 유지

### 문제 5: 테스트 프레임워크

다음을 수행하는 셸 스크립트용 테스팅 프레임워크를 구현하세요:
- 함수 단위 테스트 지원
- 외부 명령어 모킹
- 출력(stdout/stderr) 캡처 및 검증
- 종료 코드 테스트
- 어서션 제공(assert_equals, assert_contains 등)
- 커버리지 보고서 생성
- CI/CD 시스템과 통합
- JUnit 스타일 XML 보고서 생성

## 연습 문제

### 연습 1: 스크립트의 이식성 문제 감사하기

다음 스크립트 단편의 모든 이식성 문제를 식별하세요. 각 문제에 대해 문제를 설명하고 이식 가능한(portable) 수정을 제공하세요.

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

찾아야 할 문제: 배시즘(bashism), 안전하지 않은 인용(quoting), 이식 불가능한 플래그, 더 이상 사용되지 않는 문법, 특수 파일명에서 안전하지 않은 패턴.

### 연습 2: POSIX 호환 스크립트 작성하기

다음 bash 전용 스크립트를 POSIX 호환(`#!/bin/sh`)으로 다시 작성하세요. 모든 bash 전용 기능을 POSIX 동등물로 교체하세요.

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

다시 작성한 후, `bash`와 `sh` 모두로 실행되는지 테스트하세요.

### 연습 3: ShellCheck를 워크플로우에 통합하기

작은 프로젝트에 ShellCheck를 설정하세요:
1. ShellCheck 설치 (아직 사용 가능하지 않은 경우)
2. `shell=bash`를 설정하고 `SC2034`(미사용 변수 — 소스된 라이브러리에서 흔함)를 비활성화하는 `.shellcheckrc` 파일 생성
3. 현재 디렉토리 아래의 모든 `*.sh` 파일을 찾아 각각에 `shellcheck`를 실행하고, 에러(심각도(severity) `error`) 발견 시 코드 1로 종료하는 `lint.sh` 스크립트 작성
4. 테스트 스크립트에 의도적으로 두 개의 ShellCheck 경고를 도입하고 `lint.sh`가 이를 잡아내는지 확인

### 연습 4: 성능 모범 사례 적용하기

다음 느린 스크립트를 프로파일링하고 최적화하세요. `time`을 사용하여 각 변경 전후의 실행 시간을 측정하세요.

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

최소 세 가지 최적화를 적용하세요 (힌트: 루프에서 서브쉘(subshell) 피하기, 내장 문자열 연산 사용, 카운팅에는 `grep -c` 선호). 최종 `time` 출력을 원본과 비교하세요.

### 연습 5: 배포 가능한 스크립트 패키지 만들기

모범 사례를 따라 배포용 스크립트를 패키지화하세요:
- `myscript.sh`를 `/usr/local/bin`에 복사하고, 권한을 `755`로 설정하고, `/usr/local/share/man/man1/myscript.1`에 맨 페이지(man page) 항목을 생성하는 `install.sh` 작성
- 루트가 아닌 설치를 지원하는 `--prefix` 옵션 추가 (예: `~/.local`)
- 설치를 완전히 되돌리는 `uninstall.sh` 작성
- 버전 확인 추가: 시스템의 bash가 4.0보다 오래된 경우, 경고를 출력하고 종료

---

**이전**: [11_Argument_Parsing.md](./11_Argument_Parsing.md) | **다음**: [13_Testing.md](./13_Testing.md)
