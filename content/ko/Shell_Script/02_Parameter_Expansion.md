# 매개변수 확장과 변수 속성 ⭐⭐

**이전**: [Shell 기초와 실행 환경](./01_Shell_Fundamentals.md) | **다음**: [배열과 데이터 구조](./03_Arrays_and_Data.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 문자열 제거 연산자(`#`, `##`, `%`, `%%`)를 활용하여 경로 구성 요소와 파일 확장자를 추출할 수 있습니다
2. 검색 및 교체 확장(`/`, `//`)으로 문자열을 인플레이스(in-place) 변환할 수 있습니다
3. 외부 명령어 없이 부분 문자열 추출과 길이 조회를 구현할 수 있습니다
4. 설정되지 않거나 빈 변수에 대한 기본값, 대체값, 오류값을 구성할 수 있습니다
5. 대소문자 변환 연산자로 이식성 있는 문자열 정규화를 적용할 수 있습니다
6. 간접 변수 참조(`${!var}`)를 이용하여 동적 변수 조회를 수행할 수 있습니다
7. `declare`로 변수 속성(정수, 읽기 전용, nameref, 내보내기)을 설정할 수 있습니다
8. 매개변수 확장 기법을 조합하여 fork 없는 효율적인 텍스트 처리를 구축할 수 있습니다

---

매개변수 확장(Parameter Expansion)은 bash에서 가장 활용도가 낮은 기능 중 하나입니다. 단순한 문자열 조작을 위해 `sed`, `awk`, `cut` 같은 서브프로세스를 생성하는 대신, 매개변수 확장을 사용하면 인라인으로 처리할 수 있어 스크립트가 더 빠르고 읽기 쉬워집니다. 파일 경로 파싱, 설정값 처리, 입력 유효성 검사, 성능이 중요한 반복문에서의 동적 문자열 생성 등에서 이 기법이 필요합니다.

## 1. 문자열 제거 연산자

매개변수 확장은 문자열의 시작 또는 끝에서 패턴을 제거하는 내장 연산자를 제공합니다. 이는 `sed`나 외부 도구를 사용하는 것보다 빠릅니다.

### 시작 부분에서 제거 (# and ##)

```bash
#!/usr/bin/env bash

# # removes shortest match from start
# ## removes longest match from start

filepath="/usr/local/bin/script.sh"

# Remove shortest match from start
echo "${filepath#*/}"      # usr/local/bin/script.sh
echo "${filepath#*/*/}"    # local/bin/script.sh

# Remove longest match from start
echo "${filepath##*/}"     # script.sh (basename)
echo "${filepath##*.}"     # sh (extension)

# Practical: extract filename from path
filename="${filepath##*/}"
echo "Filename: $filename"

# Remove path, keep filename
url="https://example.com/path/to/file.tar.gz"
file="${url##*/}"
echo "File: $file"  # file.tar.gz
```

### 끝 부분에서 제거 (% and %%)

```bash
#!/usr/bin/env bash

# % removes shortest match from end
# %% removes longest match from end

filepath="/usr/local/bin/script.sh"

# Remove shortest match from end
echo "${filepath%/*}"      # /usr/local/bin (dirname)
echo "${filepath%.*}"      # /usr/local/bin/script (remove extension)

# Remove longest match from end
echo "${filepath%%/*}"     # (empty, removes everything)
echo "${filepath%%.*}"     # /usr/local/bin/script

# Practical: get directory from path
directory="${filepath%/*}"
echo "Directory: $directory"

# Remove extension
filename="archive.tar.gz"
base="${filename%.*}"      # archive.tar
base="${filename%%.*}"     # archive (remove all extensions)
echo "Base: $base"
```

### 비교 표

| 연산자 | 방향 | 매칭 | 예제 | 결과 |
|----------|-----------|-------|---------|--------|
| `${var#pattern}` | 시작부터 | 최단 | `${path#*/}` | 첫 디렉터리 제거 |
| `${var##pattern}` | 시작부터 | 최장 | `${path##*/}` | Basename |
| `${var%pattern}` | 끝부터 | 최단 | `${file%.*}` | 확장자 제거 |
| `${var%%pattern}` | 끝부터 | 최장 | `${file%%.*}` | 모든 확장자 제거 |

### 실용 예제

```bash
#!/usr/bin/env bash

# Extract components from URLs
url="https://user@example.com:8080/path/to/resource.html?query=1#anchor"

# Remove protocol
no_protocol="${url#*://}"
echo "No protocol: $no_protocol"
# user@example.com:8080/path/to/resource.html?query=1#anchor

# Extract domain
temp="${url#*://}"
domain="${temp%%/*}"
echo "Domain: $domain"  # user@example.com:8080

# Extract path
temp="${url#*://}"
temp="${temp#*/}"
path="/${temp%%\?*}"
echo "Path: $path"  # /path/to/resource.html

# Remove query string and anchor
clean_url="${url%%\?*}"
clean_url="${clean_url%%#*}"
echo "Clean URL: $clean_url"
# https://user@example.com:8080/path/to/resource.html

# Extract filename without extension from path
fullpath="/var/log/nginx/access.log.2024-02-13"
filename="${fullpath##*/}"      # access.log.2024-02-13
basename="${filename%%.*}"      # access
extension="${filename#*.}"      # log.2024-02-13
first_ext="${filename##*.}"     # 2024-02-13

echo "File: $filename"
echo "Base: $basename"
echo "Full ext: $extension"
echo "Last ext: $first_ext"
```

### 일괄 파일 처리

```bash
#!/usr/bin/env bash

# Remove extensions from multiple files
for file in *.tar.gz; do
    base="${file%.tar.gz}"
    echo "Extracting $file to $base/"
    mkdir -p "$base"
    tar xzf "$file" -C "$base"
done

# Convert file extensions
shopt -s nullglob
for file in *.jpeg; do
    newname="${file%.jpeg}.jpg"
    mv -v "$file" "$newname"
done

# Clean up numbered backups
for file in *.txt.{1..10}; do
    [ -f "$file" ] || continue
    original="${file%.*}"  # Remove .1, .2, etc.
    echo "Backup: $file -> original: $original"
done
```

## 2. 검색 및 치환

매개변수 확장은 패턴 검색과 치환을 지원하며, 간단한 문자열 연산을 위한 `sed`의 경량 대안을 제공합니다.

### 기본 검색 및 치환

```bash
#!/usr/bin/env bash

# / replaces first occurrence
# // replaces all occurrences

text="Hello World World World"

# Replace first occurrence
echo "${text/World/Bash}"     # Hello Bash World World

# Replace all occurrences
echo "${text//World/Bash}"    # Hello Bash Bash Bash

# Replace at start (prefix with #)
text="prefixSuffixprefix"
echo "${text/#prefix/START}"  # STARTSuffixprefix

# Replace at end (prefix with %)
echo "${text/%prefix/END}"    # prefixSuffixEND

# Delete pattern (replace with empty)
echo "${text//prefix/}"       # Suffix
```

### 패턴 매칭

```bash
#!/usr/bin/env bash

# Use glob patterns in search
path="/usr/local/bin:/usr/bin:/bin"

# Replace first path separator
echo "${path/:/ : }"          # /usr/local/bin : /usr/bin:/bin

# Replace all path separators
echo "${path//:/ : }"         # /usr/local/bin : /usr/bin : /bin

# Remove all digits
version="bash-5.1.16-release"
echo "${version//[0-9]/}"     # bash-..-release

# Remove all non-alphanumeric
string="Hello, World! 123"
echo "${string//[^a-zA-Z0-9]/}"  # HelloWorld123
```

### 실용 응용

```bash
#!/usr/bin/env bash

# Sanitize filenames
sanitize_filename() {
    local filename="$1"

    # Replace spaces with underscores
    filename="${filename// /_}"

    # Remove special characters
    filename="${filename//[^a-zA-Z0-9._-]/}"

    # Replace multiple underscores with single
    while [[ "$filename" =~ __ ]]; do
        filename="${filename//__/_}"
    done

    echo "$filename"
}

echo "$(sanitize_filename 'My Document (draft).txt')"
# My_Document_draft.txt

# Convert between separators
csv_to_tsv() {
    local line="$1"
    echo "${line//,/$'\t'}"
}

tsv_to_csv() {
    local line="$1"
    echo "${line//$'\t'/,}"
}

data="name,age,city"
echo "CSV: $data"
tsv="$(csv_to_tsv "$data")"
echo "TSV: $tsv"
echo "Back to CSV: $(tsv_to_csv "$tsv")"

# URL encoding (basic)
urlencode() {
    local string="$1"
    string="${string// /%20}"
    string="${string//&/%26}"
    string="${string//=/%3D}"
    string="${string//\?/%3F}"
    echo "$string"
}

url="search?q=hello world&lang=en"
echo "Encoded: $(urlencode "$url")"
# search%3Fq%3Dhello%20world%26lang%3Den
```

### 일괄 이름 변경

```bash
#!/usr/bin/env bash

# Rename files by replacing patterns
batch_rename() {
    local pattern="$1"
    local replacement="$2"
    local extension="${3:-*}"

    shopt -s nullglob
    for file in *."$extension"; do
        # Skip if pattern not found
        [[ "$file" != *"$pattern"* ]] && continue

        # Generate new name
        newname="${file//$pattern/$replacement}"

        # Rename if new name is different
        if [[ "$file" != "$newname" ]]; then
            echo "Renaming: $file -> $newname"
            mv -n "$file" "$newname"
        fi
    done
}

# Usage examples:
# batch_rename "draft" "final" "txt"
# batch_rename " " "_" "md"
# batch_rename "2024" "2025" "*"

# Remove date stamps from filenames
remove_datestamp() {
    shopt -s nullglob
    for file in *_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].*; do
        # Remove pattern _YYYY-MM-DD
        newname="${file/_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/}"
        echo "Renaming: $file -> $newname"
        mv -n "$file" "$newname"
    done
}

# Lowercase all extensions
lowercase_extensions() {
    shopt -s nullglob
    for file in *.*; do
        name="${file%.*}"
        ext="${file##*.}"
        newext="${ext,,}"  # Lowercase (see section 5)

        if [[ "$ext" != "$newext" ]]; then
            echo "Renaming: $file -> $name.$newext"
            mv -n "$file" "$name.$newext"
        fi
    done
}
```

## 3. 부분 문자열 및 길이 연산

매개변수 확장을 사용하여 문자열의 일부를 추출하고 문자열 길이를 결정합니다.

### 문자열 길이

```bash
#!/usr/bin/env bash

# ${#var} returns string length

string="Hello, World!"
echo "Length: ${#string}"  # 13

# Length of empty string
empty=""
echo "Empty length: ${#empty}"  # 0

# Validate input length
validate_password() {
    local password="$1"
    local min_length=8
    local max_length=64

    if [ ${#password} -lt $min_length ]; then
        echo "Password too short (min: $min_length)" >&2
        return 1
    fi

    if [ ${#password} -gt $max_length ]; then
        echo "Password too long (max: $max_length)" >&2
        return 1
    fi

    echo "Password length OK: ${#password} characters"
    return 0
}

validate_password "abc"      # Too short
validate_password "SecureP@ssw0rd"  # OK
```

### 부분 문자열 추출

```bash
#!/usr/bin/env bash

# ${var:offset:length}
# Offset is 0-indexed
# Negative offset counts from end (bash 4.2+)

string="Hello, World!"

# Extract from position 0, length 5
echo "${string:0:5}"      # Hello

# Extract from position 7 to end
echo "${string:7}"        # World!

# Extract last 6 characters (negative offset)
echo "${string: -6}"      # World!
echo "${string:(-6)}"     # World! (alternative syntax)

# Extract 5 characters starting 6 from end
echo "${string: -6:5}"    # World

# Practical: extract date components
date="2024-02-13"
year="${date:0:4}"
month="${date:5:2}"
day="${date:8:2}"

echo "Year: $year, Month: $month, Day: $day"
# Year: 2024, Month: 02, Day: 13

# Extract time components
timestamp="2024-02-13T14:30:45Z"
time="${timestamp:11:8}"
echo "Time: $time"  # 14:30:45

# Truncate long strings
truncate() {
    local string="$1"
    local max_length="${2:-50}"

    if [ ${#string} -le $max_length ]; then
        echo "$string"
    else
        echo "${string:0:$max_length}..."
    fi
}

long_text="This is a very long string that needs to be truncated for display purposes."
echo "$(truncate "$long_text" 30)"
# This is a very long string th...
```

### 문자열 패딩

```bash
#!/usr/bin/env bash

# Pad string to specific width
pad_left() {
    local string="$1"
    local width="$2"
    local padchar="${3:- }"  # Default to space

    local len=${#string}
    if [ $len -ge $width ]; then
        echo "$string"
        return
    fi

    local padding_needed=$((width - len))
    printf "%${padding_needed}s%s" "" "$string" | tr ' ' "$padchar"
}

pad_right() {
    local string="$1"
    local width="$2"
    local padchar="${3:- }"

    local len=${#string}
    if [ $len -ge $width ]; then
        echo "$string"
        return
    fi

    printf "%-${width}s" "$string" | tr ' ' "$padchar"
}

# Format table
echo "$(pad_right 'Name' 20) $(pad_left 'Value' 10)"
echo "$(pad_right '----' 20 '-') $(pad_left '-----' 10 '-')"
echo "$(pad_right 'CPU Usage' 20) $(pad_left '45%' 10)"
echo "$(pad_right 'Memory' 20) $(pad_left '2.3 GB' 10)"
echo "$(pad_right 'Disk' 20) $(pad_left '123 GB' 10)"

# Zero-pad numbers
zero_pad() {
    local number="$1"
    local width="${2:-3}"
    printf "%0${width}d" "$number"
}

echo "File_$(zero_pad 5 3).txt"     # File_005.txt
echo "File_$(zero_pad 42 4).txt"    # File_0042.txt
```

### 실용 문자열 처리

```bash
#!/usr/bin/env bash

# Parse log timestamps
parse_log_line() {
    local line="$1"
    # Format: [2024-02-13 14:30:45] INFO: Message

    # Extract timestamp (chars 1-19)
    local timestamp="${line:1:19}"

    # Extract level (after timestamp + 2 chars for '] ')
    local rest="${line:22}"
    local level="${rest%%:*}"

    # Extract message (after level + ': ')
    local message="${rest#*: }"

    echo "Timestamp: $timestamp"
    echo "Level: $level"
    echo "Message: $message"
}

log_line="[2024-02-13 14:30:45] ERROR: Connection timeout"
parse_log_line "$log_line"

# Credit card masking
mask_credit_card() {
    local cc="$1"
    # Show only last 4 digits
    local masked="${cc:0:${#cc}-4}"
    masked="${masked//[0-9]/X}"
    local visible="${cc: -4}"
    echo "${masked}${visible}"
}

echo "$(mask_credit_card '1234567890123456')"
# XXXXXXXXXXXX3456

# Extract domain from email
extract_domain() {
    local email="$1"
    echo "${email#*@}"
}

extract_username() {
    local email="$1"
    echo "${email%@*}"
}

email="user@example.com"
echo "Username: $(extract_username "$email")"
echo "Domain: $(extract_domain "$email")"
```

## 4. 기본값과 대체값

매개변수 확장은 정의되지 않았거나 비어있는 변수를 우아하게 처리하는 연산자를 제공합니다.

### 기본값 연산자

```bash
#!/usr/bin/env bash

# ${var:-default}  Use default if unset or empty
# ${var-default}   Use default if unset (not if empty)
# ${var:=default}  Assign and use default if unset or empty
# ${var=default}   Assign and use default if unset (not if empty)
# ${var:+alternate} Use alternate if set and not empty
# ${var+alternate}  Use alternate if set (even if empty)
# ${var:?error}    Error if unset or empty
# ${var?error}     Error if unset (not if empty)

# :- Use default value
unset name
echo "${name:-Anonymous}"  # Anonymous (name still unset)
echo "$name"               # (empty)

name=""
echo "${name:-Anonymous}"  # Anonymous (empty treated as unset)

name="John"
echo "${name:-Anonymous}"  # John

# - Use default only if truly unset
unset value
echo "${value-default}"    # default
value=""
echo "${value-default}"    # (empty string, not default)

# := Assign default value
unset port
echo "${port:=8080}"       # 8080
echo "$port"               # 8080 (now assigned)

# :+ Use alternate value if set
unset debug
echo "Debug: ${debug:+enabled}"   # Debug: (empty)

debug="1"
echo "Debug: ${debug:+enabled}"   # Debug: enabled

# :? Error if unset
check_required() {
    local file="${1:?Error: filename required}"
    echo "Processing: $file"
}

# check_required           # Error: filename required
check_required "data.txt"  # Processing: data.txt
```

### 기본값을 사용한 설정

```bash
#!/usr/bin/env bash

# Configuration pattern with environment variables and defaults
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-myapp}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:?Error: DB_PASSWORD must be set}"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_FILE="${LOG_FILE:-/var/log/app.log}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-100}"
TIMEOUT="${TIMEOUT:-30}"

cat <<EOF
Database Configuration:
  Host: $DB_HOST
  Port: $DB_PORT
  Database: $DB_NAME
  User: $DB_USER

Application Settings:
  Log Level: $LOG_LEVEL
  Log File: $LOG_FILE
  Max Connections: $MAX_CONNECTIONS
  Timeout: ${TIMEOUT}s
EOF

# Connect to database
connect_db() {
    psql -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -U "$DB_USER"
}

# Usage with environment variables:
# DB_PASSWORD=secret ./script.sh
# DB_HOST=prod-db DB_PORT=5433 DB_PASSWORD=secret ./script.sh
```

### 선택적 기능

```bash
#!/usr/bin/env bash

# Enable optional features based on environment
VERBOSE="${VERBOSE:-}"
DEBUG="${DEBUG:-}"
DRY_RUN="${DRY_RUN:-}"

log_verbose() {
    [ -n "$VERBOSE" ] && echo "[VERBOSE] $*" >&2
}

log_debug() {
    [ -n "$DEBUG" ] && echo "[DEBUG] $*" >&2
}

execute() {
    local cmd="$*"

    log_debug "Command: $cmd"

    if [ -n "$DRY_RUN" ]; then
        echo "[DRY-RUN] Would execute: $cmd"
        return 0
    fi

    "$@"
}

# Main logic
log_verbose "Starting process..."
log_debug "Working directory: $(pwd)"

execute rm -f temp.txt
execute mkdir -p output

log_verbose "Process complete"

# Usage:
# ./script.sh                           # Normal mode
# VERBOSE=1 ./script.sh                 # Verbose mode
# DEBUG=1 ./script.sh                   # Debug mode
# DRY_RUN=1 ./script.sh                 # Dry-run mode
# VERBOSE=1 DEBUG=1 DRY_RUN=1 ./script.sh  # All flags
```

### 필수 변수 패턴

```bash
#!/usr/bin/env bash

# Check multiple required variables
check_required() {
    local missing=()

    for var in "$@"; do
        if [ -z "${!var}" ]; then
            missing+=("$var")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo "Error: Required variables not set:" >&2
        printf '  %s\n' "${missing[@]}" >&2
        return 1
    fi

    return 0
}

# Define required variables
REQUIRED_VARS=(
    API_KEY
    API_SECRET
    ENDPOINT_URL
)

# Check all at once
if ! check_required "${REQUIRED_VARS[@]}"; then
    echo "Please set all required environment variables" >&2
    exit 1
fi

# Or check individually with custom errors
API_KEY="${API_KEY:?Error: API_KEY is required. Get one at https://example.com/api}"
API_SECRET="${API_SECRET:?Error: API_SECRET is required}"
ENDPOINT_URL="${ENDPOINT_URL:?Error: ENDPOINT_URL is required}"

echo "Configuration valid, proceeding..."
```

## 5. 대소문자 변환

Bash 4.0+는 대소문자 변환을 위한 매개변수 확장 연산자를 제공합니다.

### 대소문자 변환 연산자

```bash
#!/usr/bin/env bash

# ^   Uppercase first character
# ^^  Uppercase all characters
# ,   Lowercase first character
# ,,  Lowercase all characters

string="Hello World"

# Uppercase operations
echo "${string^}"      # Hello World (first char of first word)
echo "${string^^}"     # HELLO WORLD (all chars)

# Lowercase operations
echo "${string,}"      # hello World (first char)
echo "${string,,}"     # hello world (all chars)

# Pattern-based conversion (bash 4.0+)
string="hello world"
echo "${string^^[hw]}"  # Hello World (uppercase h and w)

string="HELLO WORLD"
echo "${string,,[HW]}"  # hELLO wORLD (lowercase H and W)
```

### 입력 정규화

```bash
#!/usr/bin/env bash

# Normalize user input to lowercase
normalize_input() {
    local input="$1"
    echo "${input,,}"
}

# Case-insensitive comparison
read -p "Continue? (yes/no): " answer
answer="$(normalize_input "$answer")"

case "$answer" in
    yes|y)
        echo "Continuing..."
        ;;
    no|n)
        echo "Aborting..."
        exit 0
        ;;
    *)
        echo "Invalid input: $answer" >&2
        exit 1
        ;;
esac

# Normalize file extensions
process_files() {
    shopt -s nullglob

    for file in *; do
        [ -f "$file" ] || continue

        # Get extension in lowercase
        ext="${file##*.}"
        ext_lower="${ext,,}"

        case "$ext_lower" in
            jpg|jpeg|png|gif)
                echo "Image: $file"
                ;;
            mp4|avi|mkv)
                echo "Video: $file"
                ;;
            txt|md|log)
                echo "Text: $file"
                ;;
            *)
                echo "Other: $file"
                ;;
        esac
    done
}
```

### 제목 대소문자

```bash
#!/usr/bin/env bash

# Convert to title case (capitalize first letter of each word)
to_title_case() {
    local string="$1"
    local result=""
    local word

    # Convert to lowercase first
    string="${string,,}"

    # Process each word
    for word in $string; do
        # Capitalize first letter
        result+="${word^} "
    done

    # Remove trailing space
    echo "${result% }"
}

echo "$(to_title_case 'hello world from bash')"
# Hello World From Bash

echo "$(to_title_case 'THE QUICK BROWN FOX')"
# The Quick Brown Fox

# Sentence case (first letter of first word only)
to_sentence_case() {
    local string="$1"
    string="${string,,}"
    echo "${string^}"
}

echo "$(to_sentence_case 'HELLO WORLD')"
# Hello world
```

### 실용 예제

```bash
#!/usr/bin/env bash

# Environment variable name normalization
normalize_env_var() {
    local name="$1"
    # Convert to uppercase and replace invalid chars with underscore
    name="${name^^}"
    name="${name//[^A-Z0-9_]/}"
    echo "$name"
}

echo "$(normalize_env_var 'my-app.config.port')"
# MY_APP_CONFIG_PORT

# SQL identifier quoting
quote_sql_identifier() {
    local identifier="$1"
    # Convert to lowercase (PostgreSQL convention)
    identifier="${identifier,,}"
    echo "\"$identifier\""
}

echo "SELECT * FROM $(quote_sql_identifier 'UserData')"
# SELECT * FROM "userdata"

# HTTP header normalization
normalize_http_header() {
    local header="$1"
    # Title-Case-With-Dashes

    local IFS='-'
    local words=($header)
    local result=""

    for word in "${words[@]}"; do
        word="${word,,}"
        result+="${word^}-"
    done

    echo "${result%-}"
}

echo "$(normalize_http_header 'content-type')"      # Content-Type
echo "$(normalize_http_header 'X-REQUEST-ID')"      # X-Request-Id
echo "$(normalize_http_header 'accept-encoding')"   # Accept-Encoding
```

## 6. 간접 참조

간접 참조(Indirect References)를 사용하면 한 변수의 값을 다른 변수의 이름으로 사용할 수 있습니다.

### 기본 간접 참조

```bash
#!/usr/bin/env bash

# ${!var} - indirect expansion

# Direct access
name="John"
echo "$name"      # John

# Indirect access
var_name="name"
echo "${!var_name}"  # John (value of $name)

# Set value indirectly using declare
declare "$var_name=Jane"
echo "$name"      # Jane

# More complex example
DB_HOST_DEV="localhost"
DB_HOST_PROD="db.example.com"
DB_HOST_STAGING="staging-db.example.com"

environment="PROD"
var_name="DB_HOST_${environment}"
echo "Connecting to: ${!var_name}"  # db.example.com
```

### 변수 이름 확장

```bash
#!/usr/bin/env bash

# ${!prefix*} - expands to names of variables starting with prefix
# ${!prefix@} - same but quoted (safer)

# Set multiple related variables
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="myapp"
DB_USER="admin"

# Get all DB_* variable names
echo "Database configuration variables:"
for var_name in ${!DB_*}; do
    echo "  $var_name = ${!var_name}"
done

# Output:
# Database configuration variables:
#   DB_HOST = localhost
#   DB_NAME = myapp
#   DB_PORT = 5432
#   DB_USER = admin

# Safer version with @ (properly quoted)
for var_name in "${!DB_@}"; do
    echo "  $var_name = ${!var_name}"
done
```

### 동적 설정

```bash
#!/usr/bin/env bash

# Load configuration for specific environment
load_config() {
    local env="${1:-dev}"
    env="${env^^}"  # Uppercase

    # Define configs for each environment
    CONFIG_HOST_DEV="localhost"
    CONFIG_PORT_DEV="8080"
    CONFIG_DB_DEV="dev_database"

    CONFIG_HOST_PROD="prod.example.com"
    CONFIG_PORT_PROD="443"
    CONFIG_DB_PROD="prod_database"

    CONFIG_HOST_STAGING="staging.example.com"
    CONFIG_PORT_STAGING="8443"
    CONFIG_DB_STAGING="staging_database"

    # Load variables for selected environment
    for var in ${!CONFIG_*}; do
        if [[ "$var" == *_${env} ]]; then
            # Extract base name (CONFIG_HOST_DEV -> CONFIG_HOST)
            local base_name="${var%_*}"
            # Set variable without environment suffix
            declare -g "$base_name=${!var}"
        fi
    done
}

# Load production config
load_config "prod"

echo "Host: $CONFIG_HOST"
echo "Port: $CONFIG_PORT"
echo "Database: $CONFIG_DB"

# Load dev config
load_config "dev"

echo "Host: $CONFIG_HOST"
echo "Port: $CONFIG_PORT"
echo "Database: $CONFIG_DB"
```

### 기능 플래그

```bash
#!/usr/bin/env bash

# Feature flag system
FEATURE_NEW_UI="enabled"
FEATURE_BETA_API="disabled"
FEATURE_ANALYTICS="enabled"
FEATURE_DARK_MODE="enabled"

is_feature_enabled() {
    local feature="$1"
    local var_name="FEATURE_${feature^^}"
    local status="${!var_name:-disabled}"

    [ "$status" = "enabled" ]
}

# Usage
if is_feature_enabled "new_ui"; then
    echo "Loading new UI..."
else
    echo "Loading classic UI..."
fi

if is_feature_enabled "dark_mode"; then
    echo "Dark mode: ON"
fi

# List all features
echo "Feature flags:"
for var in ${!FEATURE_*}; do
    feature="${var#FEATURE_}"
    status="${!var}"
    echo "  $feature: $status"
done
```

### 다중 환경 비밀 정보

```bash
#!/usr/bin/env bash

# Store secrets for multiple environments
SECRET_API_KEY_DEV="dev-key-12345"
SECRET_API_KEY_STAGING="staging-key-67890"
SECRET_API_KEY_PROD="prod-key-abcdef"

SECRET_DB_PASSWORD_DEV="dev-password"
SECRET_DB_PASSWORD_STAGING="staging-password"
SECRET_DB_PASSWORD_PROD="prod-password"

get_secret() {
    local secret_name="$1"
    local environment="${2:-dev}"
    environment="${environment^^}"

    local var_name="SECRET_${secret_name^^}_${environment}"
    local secret="${!var_name}"

    if [ -z "$secret" ]; then
        echo "Error: Secret $secret_name not found for $environment" >&2
        return 1
    fi

    echo "$secret"
}

# Usage
ENVIRONMENT="prod"
API_KEY="$(get_secret "api_key" "$ENVIRONMENT")"
DB_PASSWORD="$(get_secret "db_password" "$ENVIRONMENT")"

echo "API Key: ${API_KEY:0:10}..." # Show only first 10 chars
echo "Password: ${DB_PASSWORD:0:5}..."
```

## 7. declare 내장 명령어와 변수 속성

`declare` 내장 명령어는 변수 속성을 설정하고 변수 동작을 제어합니다.

### declare 플래그

```bash
#!/usr/bin/env bash

# declare -flag variable=value

# -i: Integer attribute
declare -i count=0
count=count+1        # Arithmetic without $(( ))
echo "$count"        # 1

count="5 + 3"
echo "$count"        # 8 (evaluated as arithmetic)

count="hello"        # Invalid, becomes 0
echo "$count"        # 0

# -r: Readonly (const)
declare -r PI=3.14159
# PI=3.14  # Error: PI: readonly variable

# -l: Lowercase
declare -l lowercase="HELLO WORLD"
echo "$lowercase"    # hello world
lowercase="MIXED CaSe"
echo "$lowercase"    # mixed case

# -u: Uppercase
declare -u uppercase="hello world"
echo "$uppercase"    # HELLO WORLD
uppercase="Mixed CaSe"
echo "$uppercase"    # MIXED CASE

# -n: Nameref (reference to another variable)
name="John"
declare -n name_ref=name
echo "$name_ref"     # John
name_ref="Jane"
echo "$name"         # Jane

# -a: Indexed array
declare -a array=(one two three)
echo "${array[0]}"   # one

# -A: Associative array
declare -A config=([host]=localhost [port]=8080)
echo "${config[host]}"  # localhost

# -x: Export (make available to child processes)
declare -x ENVIRONMENT="production"
bash -c 'echo $ENVIRONMENT'  # production

# -p: Print variable definition
declare -p PI
# declare -r PI="3.14159"
```

### Declare 플래그 비교

| 플래그 | 설명 | 예제 |
|------|-------------|---------|
| `-i` | 정수(Integer) | `declare -i count=5` |
| `-r` | 읽기 전용(Readonly) | `declare -r CONST=100` |
| `-l` | 소문자(Lowercase) | `declare -l name="JOHN"` |
| `-u` | 대문자(Uppercase) | `declare -u name="john"` |
| `-n` | 이름 참조(Nameref) | `declare -n ref=var` |
| `-a` | 인덱스 배열(Indexed array) | `declare -a arr=(1 2 3)` |
| `-A` | 연관 배열(Associative array) | `declare -A map=([key]=val)` |
| `-x` | 내보내기(Export) | `declare -x VAR=value` |
| `-g` | 전역(Global, 함수 내) | `declare -g GLOBAL=1` |
| `-p` | 선언 출력(Print declaration) | `declare -p VAR` |
| `-f` | 함수 이름(Function names) | `declare -f func_name` |
| `-F` | 함수 이름만(Function names only) | `declare -F` |

### 정수 변수

```bash
#!/usr/bin/env bash

# Integer variables auto-evaluate arithmetic
declare -i counter=0
declare -i result

# Arithmetic without $(( ))
counter+=1
echo "Counter: $counter"  # 1

counter=counter+10
echo "Counter: $counter"  # 11

# Expressions evaluated
result=5*3+2
echo "Result: $result"    # 17

# Division
result=20/3
echo "Result: $result"    # 6 (integer division)

# Use in loops
declare -i i
for i in {1..5}; do
    echo "i = $i, i*2 = $((i*2))"
done

# Automatic base conversion
declare -i hex
hex=0xff
echo "$hex"  # 255

declare -i octal
octal=0755
echo "$octal"  # 493
```

### 읽기 전용 변수

```bash
#!/usr/bin/env bash

# Readonly variables (constants)
declare -r APP_NAME="MyApp"
declare -r VERSION="1.0.0"
declare -r AUTHOR="Your Name"

# Multiple readonly declarations
declare -r \
    MAX_CONNECTIONS=100 \
    TIMEOUT=30 \
    RETRY_COUNT=3

echo "$APP_NAME v$VERSION by $AUTHOR"

# Check if variable is readonly
if declare -p APP_NAME 2>/dev/null | grep -q 'declare -r'; then
    echo "APP_NAME is readonly"
fi

# Attempt to modify causes error
# APP_NAME="NewName"  # bash: APP_NAME: readonly variable

# Make existing variable readonly
config_file="/etc/app/config.yml"
readonly config_file
# config_file="/tmp/config"  # Error: readonly variable
```

### 대소문자 변환 속성

```bash
#!/usr/bin/env bash

# Automatic lowercase
declare -l email
email="User@EXAMPLE.COM"
echo "$email"  # user@example.com

# Automatic uppercase
declare -u env_name
env_name="production"
echo "$env_name"  # PRODUCTION

# Use in functions for input normalization
process_input() {
    declare -l normalized="$1"

    case "$normalized" in
        yes|y|true|1)
            return 0
            ;;
        no|n|false|0)
            return 1
            ;;
        *)
            echo "Invalid input: $1" >&2
            return 2
            ;;
    esac
}

process_input "YES" && echo "Confirmed"
process_input "No" && echo "This won't print"
```

### 이름 참조 변수

```bash
#!/usr/bin/env bash

# Nameref: reference to another variable
original="Hello"
declare -n reference=original

echo "$reference"    # Hello
reference="World"
echo "$original"     # World

# Use in functions to modify caller's variables
increment_var() {
    local -n var_ref=$1
    var_ref=$((var_ref + 1))
}

counter=10
increment_var counter
echo "$counter"  # 11

# Swap variables
swap() {
    local -n a=$1
    local -n b=$2
    local temp="$a"
    a="$b"
    b="$temp"
}

x=5
y=10
echo "Before: x=$x, y=$y"
swap x y
echo "After: x=$x, y=$y"
# Before: x=5, y=10
# After: x=10, y=5

# Pass arrays by reference
sum_array() {
    local -n arr=$1
    local sum=0
    local element

    for element in "${arr[@]}"; do
        sum=$((sum + element))
    done

    echo "$sum"
}

numbers=(1 2 3 4 5)
result=$(sum_array numbers)
echo "Sum: $result"  # 15
```

### 변수 검사

```bash
#!/usr/bin/env bash

# Print variable declarations
declare -i count=5
declare -r VERSION="1.0"
declare -a files=(a.txt b.txt)
declare -A config=([host]=localhost)

# Print specific variable
declare -p count
# declare -i count="5"

declare -p VERSION
# declare -r VERSION="1.0"

# Print all variables
declare -p | head -20

# Print only exported variables
declare -px

# Print all functions
declare -F

# Print specific function
declare -f sum_array

# Check variable attributes
is_readonly() {
    declare -p "$1" 2>/dev/null | grep -q 'declare -r'
}

is_integer() {
    declare -p "$1" 2>/dev/null | grep -q 'declare -i'
}

is_readonly "VERSION" && echo "VERSION is readonly"
is_integer "count" && echo "count is integer"
```

## 8. 실전 패턴

### URL 파서

```bash
#!/usr/bin/env bash

# Complete URL parser using parameter expansion
parse_url() {
    local url="$1"

    # Extract protocol
    local protocol="${url%%://*}"
    local rest="${url#*://}"

    # Extract credentials if present
    local credentials=""
    local user=""
    local password=""
    if [[ "$rest" == *@* ]]; then
        credentials="${rest%%@*}"
        rest="${rest#*@}"
        user="${credentials%%:*}"
        password="${credentials#*:}"
    fi

    # Extract host and port
    local host_port="${rest%%/*}"
    local host="${host_port%%:*}"
    local port="${host_port#*:}"
    [ "$port" = "$host" ] && port=""  # No port specified

    # Extract path
    local path_query="${rest#*/}"
    [ "$path_query" = "$rest" ] && path_query=""  # No path
    local path="/${path_query%%\?*}"
    [ "$path" = "/$path_query" ] && path="/${path_query%%#*}"

    # Extract query string
    local query=""
    if [[ "$path_query" == *\?* ]]; then
        query="${path_query#*\?}"
        query="${query%%#*}"
    fi

    # Extract fragment
    local fragment=""
    if [[ "$url" == *#* ]]; then
        fragment="${url##*#}"
    fi

    # Print results
    echo "URL: $url"
    echo "  Protocol: $protocol"
    [ -n "$user" ] && echo "  User: $user"
    [ -n "$password" ] && echo "  Password: ${password:0:3}***"
    echo "  Host: $host"
    [ -n "$port" ] && echo "  Port: $port"
    echo "  Path: $path"
    [ -n "$query" ] && echo "  Query: $query"
    [ -n "$fragment" ] && echo "  Fragment: $fragment"
}

parse_url "https://user:pass@example.com:8080/path/to/resource.html?key=value&foo=bar#section"
```

### 설정 파일 파서

```bash
#!/usr/bin/env bash

# Parse key=value config files
parse_config() {
    local config_file="$1"
    local -n config_array=$2

    [ -f "$config_file" ] || return 1

    local line key value
    while IFS= read -r line; do
        # Skip empty lines and comments
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue

        # Remove inline comments
        line="${line%%#*}"

        # Trim whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"

        # Parse key=value
        key="${line%%=*}"
        value="${line#*=}"

        # Trim key and value
        key="${key#"${key%%[![:space:]]*}"}"
        key="${key%"${key##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"

        # Remove quotes from value
        if [[ "$value" =~ ^\".*\"$ ]] || [[ "$value" =~ ^\'.*\'$ ]]; then
            value="${value:1:-1}"
        fi

        # Store in associative array
        config_array["$key"]="$value"
    done < "$config_file"
}

# Usage
declare -A config

cat > test_config.conf <<'EOF'
# Database configuration
db_host = localhost
db_port = 5432
db_name = "myapp"

# Application settings
app_name = My Application
log_level = INFO  # inline comment
timeout = 30
EOF

parse_config "test_config.conf" config

echo "Configuration loaded:"
for key in "${!config[@]}"; do
    echo "  $key = ${config[$key]}"
done

rm test_config.conf
```

### 경로 조작 도구킷

```bash
#!/usr/bin/env bash

# Complete path manipulation library

# Get absolute path
abspath() {
    local path="$1"

    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$(pwd)/$path"
    fi
}

# Get directory name (like dirname)
dirname() {
    local path="$1"

    # Remove trailing slashes
    path="${path%/}"

    # Get directory part
    local dir="${path%/*}"

    # If no directory part, return .
    [ "$dir" = "$path" ] && dir="."

    echo "$dir"
}

# Get filename (like basename)
basename() {
    local path="$1"
    local suffix="$2"

    # Remove trailing slashes
    path="${path%/}"

    # Get filename
    local name="${path##*/}"

    # Remove suffix if provided
    if [ -n "$suffix" ]; then
        name="${name%$suffix}"
    fi

    echo "$name"
}

# Get file extension
get_extension() {
    local path="$1"
    local name="${path##*/}"

    # No extension if no dot or starts with dot
    [[ "$name" != *.* ]] && return
    [[ "$name" = .* ]] && return

    echo "${name##*.}"
}

# Remove extension
remove_extension() {
    local path="$1"
    echo "${path%.*}"
}

# Replace extension
replace_extension() {
    local path="$1"
    local new_ext="$2"
    local base="${path%.*}"
    echo "${base}.${new_ext}"
}

# Join paths
join_path() {
    local IFS='/'
    local joined="$*"

    # Remove duplicate slashes
    while [[ "$joined" =~ // ]]; do
        joined="${joined//\/\//\/}"
    done

    echo "$joined"
}

# Normalize path (remove ./ and ../)
normalize_path() {
    local path="$1"
    local IFS='/'
    local parts=($path)
    local result=()

    for part in "${parts[@]}"; do
        case "$part" in
            .|'')
                continue
                ;;
            ..)
                [ ${#result[@]} -gt 0 ] && unset 'result[-1]'
                ;;
            *)
                result+=("$part")
                ;;
        esac
    done

    local normalized="${result[*]}"
    [ "${path:0:1}" = / ] && normalized="/$normalized"
    echo "$normalized"
}

# Test the toolkit
echo "=== Path Manipulation Toolkit ==="
path="/usr/local/bin/script.sh"

echo "Original: $path"
echo "Directory: $(dirname "$path")"
echo "Filename: $(basename "$path")"
echo "Extension: $(get_extension "$path")"
echo "Without ext: $(remove_extension "$path")"
echo "Replace ext: $(replace_extension "$path" "bash")"
echo "Join: $(join_path "/usr" "local" "bin" "test.sh")"
echo "Normalize: $(normalize_path "/usr/./local/../local/bin//script.sh")"
```

## 연습 문제

### 문제 1: 고급 파일 이름 처리기

다음 기능을 가진 파일 이름 처리 스크립트를 작성하세요:
- 다양한 형식의 날짜 스탬프 추출 (YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD)
- 날짜 스탬프 제거 또는 정규화
- 버전 번호 추출 (v1.0, version-2.3.4 등)
- 파일 이름 정리 (특수 문자 제거, 공백 정규화)
- 현재 날짜/버전으로 새 파일 이름 생성

**예제**:
```bash
process_filename "report_20240213_v1.3.pdf"
# Date: 2024-02-13
# Version: 1.3
# Base: report
# Extension: pdf
# Normalized: report_2024-02-13_v1.3.pdf
```

### 문제 2: 환경 변수 검증기

환경 변수를 검증하는 스크립트를 만드세요:
- 필수 변수가 설정되어 있고 비어있지 않은지 확인
- 변수 타입 검증 (정수, 부울, 열거형)
- 값 범위 및 제약 조건 확인
- 제안을 포함한 상세한 오류 메시지 제공
- .env 파일에서 로드 지원

**예제**:
```bash
validate_env PORT        --type int --range 1024-65535 --required
validate_env DEBUG       --type bool --default false
validate_env ENVIRONMENT --type enum --values "dev,staging,prod" --required
validate_env API_KEY     --type string --min-length 32 --required
```

### 문제 3: 고급 설정 파일 관리자

다음 기능을 가진 설정 파일 관리자를 작성하세요:
- 섹션이 있는 INI 스타일 설정 파일 읽기
- 여러 데이터 타입 지원 (문자열, 정수, 부울, 배열)
- 중첩 섹션 허용 (section.subsection.key)
- get/set 연산 제공
- 스키마에 대한 값 검증
- 여러 설정 파일 병합

**예제 설정**:
```ini
[database]
host = localhost
port = 5432
databases = ["app", "cache", "logs"]

[database.pool]
min_size = 5
max_size = 20
```

### 문제 4: 문자열 템플릿 엔진

다음 기능을 가진 간단한 템플릿 엔진을 만드세요:
- 변수 치환: `Hello {{name}}!`
- 기본값 지원: `{{var:default}}`
- 조건부 섹션: `{{#if var}}...{{/if}}`
- 반복: `{{#each items}}...{{/each}}`
- 필터: `{{name | uppercase}}`
- 이스케이핑: `{{!raw}}`

**예제**:
```bash
template='Hello {{name:Guest}}! {{#if premium}}Welcome premium member{{/if}}'
render "$template" name="John" premium=true
# Output: Hello John! Welcome premium member
```

### 문제 5: 경로 해결 라이브러리

다음 기능을 가진 경로 해결 라이브러리를 구현하세요:
- 상대 경로를 절대 경로로 해결
- 심볼릭 링크 처리
- 순환 심볼릭 링크 감지
- PATH에서 파일 찾기
- 변수가 포함된 경로 해결 ($HOME, ~ 등)
- 크로스 플랫폼 지원 (Windows 경로 처리)
- 안전성 검사 (디렉터리 순회 방지)

**예제**:
```bash
resolve_path "~/Documents/../Downloads/./file.txt"
# /home/user/Downloads/file.txt

find_in_path "python3"
# /usr/bin/python3

is_subpath "/var/www/html" "/var/www/html/../../../../etc/passwd"
# Error: Directory traversal detected
```

---

**이전**: [01_Shell_Fundamentals.md](./01_Shell_Fundamentals.md) | **다음**: [03_Arrays_and_Data.md](./03_Arrays_and_Data.md)
