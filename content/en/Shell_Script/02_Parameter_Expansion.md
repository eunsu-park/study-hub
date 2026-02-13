# Parameter Expansion and Variable Attributes ⭐⭐

**Previous**: [01_Shell_Fundamentals.md](./01_Shell_Fundamentals.md) | **Next**: [03_Arrays_and_Data.md](./03_Arrays_and_Data.md)

---

This lesson explores bash parameter expansion, a powerful feature that allows you to manipulate variables directly in the shell without external tools. We'll cover string operations, default values, variable attributes, and practical patterns.

## 1. String Removal Operators

Parameter expansion provides built-in operators to remove patterns from the beginning or end of strings. These are faster than using `sed` or external tools.

### Removal from Start (# and ##)

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

### Removal from End (% and %%)

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

### Comparison Table

| Operator | Direction | Match | Example | Result |
|----------|-----------|-------|---------|--------|
| `${var#pattern}` | From start | Shortest | `${path#*/}` | Remove first dir |
| `${var##pattern}` | From start | Longest | `${path##*/}` | Basename |
| `${var%pattern}` | From end | Shortest | `${file%.*}` | Remove extension |
| `${var%%pattern}` | From end | Longest | `${file%%.*}` | Remove all extensions |

### Practical Examples

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

### Batch File Processing

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

## 2. Search and Replace

Parameter expansion supports pattern search and replacement, providing a lightweight alternative to `sed` for simple string operations.

### Basic Search and Replace

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

### Pattern Matching

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

### Practical Applications

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

### Batch Renaming

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

## 3. Substring and Length Operations

Extract portions of strings and determine string lengths using parameter expansion.

### Length of String

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

### Substring Extraction

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

### Padding Strings

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

### Practical String Processing

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

## 4. Default and Alternate Values

Parameter expansion provides operators to handle undefined or empty variables gracefully.

### Default Value Operators

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

### Configuration with Defaults

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

### Optional Features

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

### Required Variables Pattern

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

## 5. Case Conversion

Bash 4.0+ provides parameter expansion operators for case conversion.

### Case Conversion Operators

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

### Normalizing Input

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

### Title Case

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

### Practical Examples

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

## 6. Indirect References

Indirect references allow you to use the value of one variable as the name of another variable.

### Basic Indirect Reference

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

### Variable Name Expansion

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

### Dynamic Configuration

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

### Feature Flags

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

### Multi-Environment Secrets

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

## 7. declare Builtin and Variable Attributes

The `declare` builtin sets variable attributes and controls variable behavior.

### declare Flags

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

### Declare Flags Comparison

| Flag | Description | Example |
|------|-------------|---------|
| `-i` | Integer | `declare -i count=5` |
| `-r` | Readonly | `declare -r CONST=100` |
| `-l` | Lowercase | `declare -l name="JOHN"` |
| `-u` | Uppercase | `declare -u name="john"` |
| `-n` | Nameref | `declare -n ref=var` |
| `-a` | Indexed array | `declare -a arr=(1 2 3)` |
| `-A` | Associative array | `declare -A map=([key]=val)` |
| `-x` | Export | `declare -x VAR=value` |
| `-g` | Global (in function) | `declare -g GLOBAL=1` |
| `-p` | Print declaration | `declare -p VAR` |
| `-f` | Function names | `declare -f func_name` |
| `-F` | Function names only | `declare -F` |

### Integer Variables

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

### Readonly Variables

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

### Case Conversion Attributes

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

### Nameref Variables

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

### Inspecting Variables

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

## 8. Real-World Patterns

### URL Parser

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

### Config File Parser

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

### Path Manipulation Toolkit

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

## Practice Problems

### Problem 1: Advanced Filename Processor

Write a script that processes filenames with the following features:
- Extract date stamps in various formats (YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD)
- Remove or normalize date stamps
- Extract version numbers (v1.0, version-2.3.4, etc.)
- Sanitize filenames (remove special characters, normalize spaces)
- Generate new filenames with current date/version

**Example**:
```bash
process_filename "report_20240213_v1.3.pdf"
# Date: 2024-02-13
# Version: 1.3
# Base: report
# Extension: pdf
# Normalized: report_2024-02-13_v1.3.pdf
```

### Problem 2: Environment Variable Validator

Create a script that validates environment variables:
- Check required variables are set and non-empty
- Validate variable types (integer, boolean, enum)
- Check value ranges and constraints
- Provide detailed error messages with suggestions
- Support loading from .env files

**Example**:
```bash
validate_env PORT        --type int --range 1024-65535 --required
validate_env DEBUG       --type bool --default false
validate_env ENVIRONMENT --type enum --values "dev,staging,prod" --required
validate_env API_KEY     --type string --min-length 32 --required
```

### Problem 3: Advanced Config File Manager

Write a config file manager that:
- Reads INI-style config files with sections
- Supports multiple data types (string, int, bool, array)
- Allows nested sections (section.subsection.key)
- Provides get/set operations
- Validates values against schema
- Merges multiple config files

**Example config**:
```ini
[database]
host = localhost
port = 5432
databases = ["app", "cache", "logs"]

[database.pool]
min_size = 5
max_size = 20
```

### Problem 4: String Template Engine

Create a simple template engine that:
- Replaces variables: `Hello {{name}}!`
- Supports defaults: `{{var:default}}`
- Conditional sections: `{{#if var}}...{{/if}}`
- Loops: `{{#each items}}...{{/each}}`
- Filters: `{{name | uppercase}}`
- Escaping: `{{!raw}}`

**Example**:
```bash
template='Hello {{name:Guest}}! {{#if premium}}Welcome premium member{{/if}}'
render "$template" name="John" premium=true
# Output: Hello John! Welcome premium member
```

### Problem 5: Path Resolution Library

Implement a path resolution library with:
- Resolve relative paths to absolute
- Handle symbolic links
- Detect circular symlinks
- Find file in PATH
- Resolve paths with variables ($HOME, ~, etc.)
- Cross-platform support (handle Windows paths)
- Safety checks (directory traversal prevention)

**Example**:
```bash
resolve_path "~/Documents/../Downloads/./file.txt"
# /home/user/Downloads/file.txt

find_in_path "python3"
# /usr/bin/python3

is_subpath "/var/www/html" "/var/www/html/../../../../etc/passwd"
# Error: Directory traversal detected
```

---

**Previous**: [01_Shell_Fundamentals.md](./01_Shell_Fundamentals.md) | **Next**: [03_Arrays_and_Data.md](./03_Arrays_and_Data.md)
