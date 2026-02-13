# Lesson 08: Regular Expressions in Bash

**Difficulty**: ⭐⭐⭐

**Previous**: [07_String_Processing.md](./07_String_Processing.md) | **Next**: [09_Process_Management.md](./09_Process_Management.md)

---

## 1. Glob vs Regex

Understanding the difference between globs and regex is crucial for bash scripting.

### 1.1 Fundamental Differences

| Feature | Glob | Regex |
|---------|------|-------|
| **Purpose** | Filename matching | String pattern matching |
| **Context** | File operations, case statements | `[[ =~ ]]`, grep, sed, awk |
| **`*` meaning** | Zero or more characters | Zero or more of previous character |
| **`.` meaning** | Literal dot | Any single character |
| **`?` meaning** | Exactly one character | Zero or one of previous character |
| **Character class** | `[abc]` | `[abc]` (same) |
| **Negation** | `[!abc]` | `[^abc]` |
| **Anchors** | None (implicit) | `^` (start), `$` (end) |
| **Groups** | `{a,b}` (brace expansion) | `(a\|b)` (alternation) |

### 1.2 Glob Examples

```bash
#!/bin/bash

# Globs are used for filename matching
ls *.txt              # All files ending in .txt
ls test?.log          # test1.log, test2.log, etc.
ls [abc]*.txt         # Files starting with a, b, or c
ls [!0-9]*            # Files not starting with digit
ls file{1,2,3}.txt    # file1.txt, file2.txt, file3.txt

# In case statements
case $filename in
    *.txt)    echo "Text file" ;;
    *.jpg|*.png) echo "Image file" ;;
    test*)    echo "Test file" ;;
esac

# In conditionals
if [[ $filename == *.txt ]]; then
    echo "Text file"
fi
```

### 1.3 Regex Examples

```bash
#!/bin/bash

# Regex is used for string matching
[[ $string =~ ^[0-9]+$ ]] && echo "All digits"
[[ $email =~ ^[a-z]+@[a-z]+\.[a-z]+$ ]] && echo "Valid email pattern"

# With grep
grep '^ERROR' logfile.txt        # Lines starting with ERROR
grep 'test.*done' logfile.txt    # 'test' followed by 'done'

# With sed
sed 's/[0-9]\+/NUM/g' file.txt   # Replace numbers with NUM
```

### 1.4 Common Confusion Points

```bash
#!/bin/bash

# WRONG: Using glob pattern with =~
[[ $str =~ *.txt ]]  # This matches literal "*" followed by ".txt"!

# CORRECT: Use glob with ==
[[ $str == *.txt ]]  # Matches strings ending with .txt

# CORRECT: Use regex with =~
[[ $str =~ .*\.txt$ ]]  # Regex: anything ending with .txt

# Glob: * means zero or more characters
echo test* matches: test, test1, test123

# Regex: * means zero or more of previous character
[[ $str =~ test* ]]  # Matches: tes, test, testt, testtt

# Glob: ? means exactly one character
ls file?.txt  # Matches: file1.txt, fileA.txt

# Regex: ? means zero or one of previous character
[[ $str =~ tests? ]]  # Matches: test, tests
```

## 2. The =~ Operator

The `=~` operator performs regex matching in bash's `[[ ]]` construct.

### 2.1 Basic Usage

```bash
#!/bin/bash

# Simple pattern matching
string="hello123"

if [[ $string =~ [0-9] ]]; then
    echo "Contains a digit"
fi

# Anchored patterns
if [[ $string =~ ^hello ]]; then
    echo "Starts with 'hello'"
fi

if [[ $string =~ [0-9]$ ]]; then
    echo "Ends with a digit"
fi

# Full string match
if [[ $string =~ ^[a-z]+[0-9]+$ ]]; then
    echo "Letters followed by numbers"
fi
```

### 2.2 Quoting Behavior

```bash
#!/bin/bash

string="test123"

# WRONG: Quoted pattern becomes literal
if [[ $string =~ "[0-9]+" ]]; then
    echo "Never matches - looks for literal '[0-9]+'"
fi

# CORRECT: Unquoted pattern
if [[ $string =~ [0-9]+ ]]; then
    echo "Matches one or more digits"
fi

# CORRECT: Variable containing pattern (quoted)
pattern='[0-9]+'
if [[ $string =~ $pattern ]]; then
    echo "Matches - variable expansion is NOT quoted"
fi

# IMPORTANT: For portability and special characters, use a variable
```

### 2.3 Return Value

```bash
#!/bin/bash

string="test123"

# Return values
if [[ $string =~ [0-9]+ ]]; then
    echo "Match found (returns 0)"
else
    echo "No match (returns 1)"
fi

# Using return value directly
validate_email() {
    [[ $1 =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]
    # Returns 0 if match, 1 if no match
}

if validate_email "user@example.com"; then
    echo "Valid email"
fi
```

### 2.4 Pattern in Variable

```bash
#!/bin/bash

# Define patterns in variables for clarity and reusability
readonly IP_PATTERN='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
readonly EMAIL_PATTERN='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
readonly URL_PATTERN='^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

validate_ip() {
    [[ $1 =~ $IP_PATTERN ]]
}

validate_email() {
    [[ $1 =~ $EMAIL_PATTERN ]]
}

validate_url() {
    [[ $1 =~ $URL_PATTERN ]]
}

# Usage
if validate_ip "192.168.1.1"; then
    echo "Valid IP"
fi
```

## 3. BASH_REMATCH

The `BASH_REMATCH` array stores captured groups from regex matches.

### 3.1 Basic Capture Groups

```bash
#!/bin/bash

string="John Doe, age 30"
pattern='([A-Z][a-z]+) ([A-Z][a-z]+), age ([0-9]+)'

if [[ $string =~ $pattern ]]; then
    echo "Full match: ${BASH_REMATCH[0]}"
    echo "First name: ${BASH_REMATCH[1]}"
    echo "Last name: ${BASH_REMATCH[2]}"
    echo "Age: ${BASH_REMATCH[3]}"
fi

# Output:
# Full match: John Doe, age 30
# First name: John
# Last name: Doe
# Age: 30
```

### 3.2 Extracting Structured Data

```bash
#!/bin/bash

# Parse log entry
log_entry="2024-02-13 14:30:45 [ERROR] Database connection failed"
pattern='^([0-9-]+) ([0-9:]+) \[([A-Z]+)\] (.+)$'

if [[ $log_entry =~ $pattern ]]; then
    date="${BASH_REMATCH[1]}"
    time="${BASH_REMATCH[2]}"
    level="${BASH_REMATCH[3]}"
    message="${BASH_REMATCH[4]}"

    echo "Date: $date"
    echo "Time: $time"
    echo "Level: $level"
    echo "Message: $message"
fi
```

### 3.3 Nested Groups

```bash
#!/bin/bash

# Parse URL
url="https://user:pass@example.com:8080/path/to/resource?key=value"
pattern='^(https?)://([^:]+):([^@]+)@([^:]+):([0-9]+)(/[^?]*)(\?.*)?$'

if [[ $url =~ $pattern ]]; then
    protocol="${BASH_REMATCH[1]}"
    username="${BASH_REMATCH[2]}"
    password="${BASH_REMATCH[3]}"
    host="${BASH_REMATCH[4]}"
    port="${BASH_REMATCH[5]}"
    path="${BASH_REMATCH[6]}"
    query="${BASH_REMATCH[7]}"

    echo "Protocol: $protocol"
    echo "Username: $username"
    echo "Password: $password"
    echo "Host: $host"
    echo "Port: $port"
    echo "Path: $path"
    echo "Query: $query"
fi
```

### 3.4 Multiple Matches in Loop

```bash
#!/bin/bash

# Extract all email addresses from text
text="Contact us at support@example.com or sales@example.com for more info."
pattern='[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

while [[ $text =~ $pattern ]]; do
    email="${BASH_REMATCH[0]}"
    echo "Found: $email"

    # Remove matched portion to find next match
    text="${text#*"$email"}"
done

# Output:
# Found: support@example.com
# Found: sales@example.com
```

### 3.5 Practical Extraction Function

```bash
#!/bin/bash

# Extract key-value pairs from string
parse_key_value() {
    local text=$1
    declare -gA parsed_data

    local pattern='([a-zA-Z_][a-zA-Z0-9_]*)=([^,]+)'

    while [[ $text =~ $pattern ]]; do
        local key="${BASH_REMATCH[1]}"
        local value="${BASH_REMATCH[2]}"

        parsed_data[$key]="$value"

        # Remove matched portion
        text="${text#*"${BASH_REMATCH[0]}"}"
    done
}

# Usage
data="name=Alice,age=30,city=NYC,role=admin"
parse_key_value "$data"

for key in "${!parsed_data[@]}"; do
    echo "$key: ${parsed_data[$key]}"
done
```

## 4. Extended Regular Expressions

ERE (Extended Regular Expressions) provide more powerful pattern matching.

### 4.1 Character Classes

```bash
#!/bin/bash

# POSIX character classes
[[ $char =~ [[:alpha:]] ]]   # Alphabetic character
[[ $char =~ [[:digit:]] ]]   # Digit
[[ $char =~ [[:alnum:]] ]]   # Alphanumeric
[[ $char =~ [[:space:]] ]]   # Whitespace
[[ $char =~ [[:punct:]] ]]   # Punctuation
[[ $char =~ [[:upper:]] ]]   # Uppercase letter
[[ $char =~ [[:lower:]] ]]   # Lowercase letter
[[ $char =~ [[:xdigit:]] ]]  # Hexadecimal digit

# Custom character classes
[[ $char =~ [aeiouAEIOU] ]]  # Vowels
[[ $char =~ [^aeiouAEIOU] ]] # Consonants (negation)
[[ $char =~ [0-9a-fA-F] ]]   # Hexadecimal character
```

### 4.2 Quantifiers

```bash
#!/bin/bash

# Basic quantifiers
[[ $str =~ a+ ]]      # One or more 'a'
[[ $str =~ a* ]]      # Zero or more 'a'
[[ $str =~ a? ]]      # Zero or one 'a'

# Bounded quantifiers
[[ $str =~ a{3} ]]    # Exactly 3 'a's
[[ $str =~ a{3,} ]]   # 3 or more 'a's
[[ $str =~ a{3,5} ]]  # Between 3 and 5 'a's

# Practical examples
[[ $str =~ ^[0-9]{3}-[0-9]{2}-[0-9]{4}$ ]]  # SSN format: 123-45-6789
[[ $str =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]  # IP address
[[ $str =~ ^[a-zA-Z]{2,20}$ ]]  # Name: 2-20 letters
```

### 4.3 Alternation

```bash
#!/bin/bash

# Alternation (OR)
[[ $str =~ ^(yes|no|maybe)$ ]]
[[ $str =~ \.(jpg|png|gif)$ ]]
[[ $str =~ ^(http|https|ftp):// ]]

# With groups
[[ $str =~ ^(Mr|Mrs|Ms|Dr)\. [A-Z][a-z]+ [A-Z][a-z]+$ ]]
# Matches: Mr. John Smith, Dr. Jane Doe, etc.

# Complex alternation
file_pattern='.*\.(txt|log|conf|cfg|ini|yaml|yml|json|xml)$'
[[ $filename =~ $file_pattern ]]
```

### 4.4 Grouping

```bash
#!/bin/bash

# Groups for capturing
pattern='(https?)://([^/]+)(/.*)?'
url="https://example.com/path/to/page"

if [[ $url =~ $pattern ]]; then
    protocol="${BASH_REMATCH[1]}"
    domain="${BASH_REMATCH[2]}"
    path="${BASH_REMATCH[3]}"
fi

# Groups for quantifiers
[[ $str =~ ^(ab)+ ]]        # Matches: ab, abab, ababab
[[ $str =~ ^([0-9]{3}-){2}[0-9]{4}$ ]]  # Phone: 555-123-4567

# Groups for alternation
[[ $str =~ ^(red|green|blue) (car|bike|boat)$ ]]
# Matches: "red car", "green bike", "blue boat", etc.
```

### 4.5 Anchors

```bash
#!/bin/bash

# Start and end anchors
[[ $str =~ ^hello ]]   # Starts with 'hello'
[[ $str =~ world$ ]]   # Ends with 'world'
[[ $str =~ ^test$ ]]   # Exactly 'test'

# Word boundaries (with grep, not directly in =~)
echo "hello world" | grep -E '\bhello\b'  # Matches 'hello' as whole word

# Practical examples
[[ $str =~ ^# ]]       # Comment line (starts with #)
[[ $str =~ ;$ ]]       # Ends with semicolon
[[ $str =~ ^$ ]]       # Empty line
[[ $str =~ ^[[:space:]]*$ ]]  # Blank line (only whitespace)
```

### 4.6 ERE vs BRE Comparison

| Feature | BRE (Basic) | ERE (Extended) |
|---------|-------------|----------------|
| Grouping | `\(\)` | `()` |
| Alternation | `\|` | `|` |
| Quantifiers `+`, `?` | Not supported | `+`, `?` |
| Bounded quantifiers | `\{n,m\}` | `{n,m}` |
| Used in | grep, sed (default) | grep -E, egrep, awk |
| Bash `=~` | Uses ERE | Native |

```bash
#!/bin/bash

# BRE (grep default)
echo "test123" | grep '\([a-z]\+\)[0-9]\+'

# ERE (grep -E or egrep)
echo "test123" | grep -E '([a-z]+)[0-9]+'

# Bash =~ uses ERE
[[ "test123" =~ ^([a-z]+)([0-9]+)$ ]]
```

## 5. Practical Validation Functions

### 5.1 Email Validation

```bash
#!/bin/bash

#
# Validate email address
#
# Returns: 0 if valid, 1 if invalid
#
validate_email() {
    local email=$1
    local pattern='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    [[ $email =~ $pattern ]]
}

# Test cases
test_email() {
    local email=$1
    if validate_email "$email"; then
        echo "✓ Valid: $email"
    else
        echo "✗ Invalid: $email"
    fi
}

test_email "user@example.com"      # ✓ Valid
test_email "user.name@example.co.uk"  # ✓ Valid
test_email "user+tag@example.com"  # ✓ Valid
test_email "invalid@"              # ✗ Invalid
test_email "@example.com"          # ✗ Invalid
test_email "user@example"          # ✗ Invalid
```

### 5.2 IPv4 Address Validation

```bash
#!/bin/bash

#
# Validate IPv4 address (with range checking)
#
validate_ipv4() {
    local ip=$1
    local pattern='^([0-9]{1,3}\.){3}[0-9]{1,3}$'

    # Check format
    if ! [[ $ip =~ $pattern ]]; then
        return 1
    fi

    # Check each octet is 0-255
    local IFS='.'
    read -ra octets <<< "$ip"

    for octet in "${octets[@]}"; do
        if ((octet < 0 || octet > 255)); then
            return 1
        fi
    done

    return 0
}

# Test cases
test_ip() {
    local ip=$1
    if validate_ipv4 "$ip"; then
        echo "✓ Valid: $ip"
    else
        echo "✗ Invalid: $ip"
    fi
}

test_ip "192.168.1.1"      # ✓ Valid
test_ip "10.0.0.1"         # ✓ Valid
test_ip "255.255.255.255"  # ✓ Valid
test_ip "256.1.1.1"        # ✗ Invalid (256 > 255)
test_ip "192.168.1"        # ✗ Invalid (incomplete)
test_ip "192.168.1.1.1"    # ✗ Invalid (too many octets)
```

### 5.3 Date Format Validation

```bash
#!/bin/bash

#
# Validate date in YYYY-MM-DD format
#
validate_date() {
    local date=$1
    local pattern='^([0-9]{4})-([0-9]{2})-([0-9]{2})$'

    if ! [[ $date =~ $pattern ]]; then
        return 1
    fi

    local year="${BASH_REMATCH[1]}"
    local month="${BASH_REMATCH[2]}"
    local day="${BASH_REMATCH[3]}"

    # Validate month
    if ((month < 1 || month > 12)); then
        return 1
    fi

    # Validate day
    if ((day < 1 || day > 31)); then
        return 1
    fi

    # Additional validation with date command
    if ! date -d "$date" > /dev/null 2>&1; then
        return 1
    fi

    return 0
}

# Test cases
test_date() {
    local date=$1
    if validate_date "$date"; then
        echo "✓ Valid: $date"
    else
        echo "✗ Invalid: $date"
    fi
}

test_date "2024-02-13"  # ✓ Valid
test_date "2024-12-31"  # ✓ Valid
test_date "2024-02-30"  # ✗ Invalid (Feb 30 doesn't exist)
test_date "2024-13-01"  # ✗ Invalid (month 13)
test_date "24-02-13"    # ✗ Invalid (wrong format)
```

### 5.4 URL Validation

```bash
#!/bin/bash

#
# Validate URL
#
validate_url() {
    local url=$1
    local pattern='^(https?|ftp)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^[:space:]]*)?$'

    [[ $url =~ $pattern ]]
}

# More comprehensive URL validation
validate_url_detailed() {
    local url=$1
    local pattern='^(https?|ftp)://(([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|localhost|([0-9]{1,3}\.){3}[0-9]{1,3})(:[0-9]{1,5})?(/[^[:space:]]*)?$'

    [[ $url =~ $pattern ]]
}

# Test cases
test_url() {
    local url=$1
    if validate_url_detailed "$url"; then
        echo "✓ Valid: $url"
    else
        echo "✗ Invalid: $url"
    fi
}

test_url "https://example.com"              # ✓ Valid
test_url "http://example.com/path/to/page"  # ✓ Valid
test_url "https://sub.example.com:8080/api" # ✓ Valid
test_url "ftp://files.example.com"          # ✓ Valid
test_url "https://localhost:3000"           # ✓ Valid
test_url "https://192.168.1.1:8080"         # ✓ Valid
test_url "htp://example.com"                # ✗ Invalid (typo in protocol)
test_url "https://example"                  # ✗ Invalid (no TLD)
```

### 5.5 Semantic Version Validation

```bash
#!/bin/bash

#
# Validate semantic version (semver)
# Format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
#
validate_semver() {
    local version=$1
    local pattern='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$'

    [[ $version =~ $pattern ]]
}

# Extract semver components
parse_semver() {
    local version=$1
    local pattern='^([0-9]+)\.([0-9]+)\.([0-9]+)(-([a-zA-Z0-9.-]+))?(\+([a-zA-Z0-9.-]+))?$'

    if [[ $version =~ $pattern ]]; then
        echo "Major: ${BASH_REMATCH[1]}"
        echo "Minor: ${BASH_REMATCH[2]}"
        echo "Patch: ${BASH_REMATCH[3]}"
        echo "Prerelease: ${BASH_REMATCH[5]}"
        echo "Build: ${BASH_REMATCH[7]}"
        return 0
    fi

    return 1
}

# Test cases
test_semver() {
    local version=$1
    if validate_semver "$version"; then
        echo "✓ Valid: $version"
        parse_semver "$version"
    else
        echo "✗ Invalid: $version"
    fi
    echo
}

test_semver "1.0.0"                    # ✓ Valid
test_semver "1.0.0-alpha"              # ✓ Valid
test_semver "1.0.0-alpha.1"            # ✓ Valid
test_semver "1.0.0+20240213"           # ✓ Valid
test_semver "1.0.0-beta+exp.sha.5114f85"  # ✓ Valid
test_semver "1.0"                      # ✗ Invalid (incomplete)
test_semver "v1.0.0"                   # ✗ Invalid (has 'v' prefix)
```

### 5.6 Comprehensive Validation Framework

```bash
#!/bin/bash

# Validation rules
declare -A VALIDATORS=(
    [email]='validate_email'
    [ipv4]='validate_ipv4'
    [date]='validate_date'
    [url]='validate_url'
    [semver]='validate_semver'
)

# Generic validation function
validate() {
    local type=$1
    local value=$2
    local validator="${VALIDATORS[$type]}"

    if [[ -z $validator ]]; then
        echo "Error: Unknown validator type: $type" >&2
        return 2
    fi

    if ! type "$validator" > /dev/null 2>&1; then
        echo "Error: Validator function not found: $validator" >&2
        return 2
    fi

    "$validator" "$value"
}

# Usage example
validate_input() {
    local field=$1
    local value=$2
    local type=$3

    if validate "$type" "$value"; then
        echo "✓ $field is valid"
        return 0
    else
        echo "✗ $field is invalid: $value" >&2
        return 1
    fi
}

# Example: validate user input
validate_input "Email" "user@example.com" "email"
validate_input "IP Address" "192.168.1.1" "ipv4"
validate_input "Version" "2.1.0-beta" "semver"
```

## 6. Regex with grep and sed

### 6.1 grep with Extended Regex

```bash
#!/bin/bash

# Extended regex with grep -E
grep -E '^[0-9]+$' file.txt          # Lines containing only digits
grep -E '(error|warning|critical)' log.txt  # Multiple patterns
grep -E '\b[A-Z]{3,}\b' file.txt     # 3+ uppercase words

# Case-insensitive
grep -iE 'error' log.txt

# Invert match
grep -vE '^#' config.txt             # Non-comment lines

# Count matches
grep -cE 'pattern' file.txt

# Show context
grep -E -A 3 -B 3 'ERROR' log.txt    # 3 lines before and after
```

### 6.2 sed Pattern Matching

```bash
#!/bin/bash

# Basic substitution with regex
sed 's/[0-9]\+/NUM/g' file.txt       # Replace numbers

# Anchored patterns
sed 's/^#.*//' file.txt              # Remove comment lines
sed 's/[[:space:]]\+$//' file.txt    # Remove trailing whitespace

# Groups and back-references
sed 's/\([0-9]\{3\}\)-\([0-9]\{2\}\)-\([0-9]\{4\}\)/(\1) \2-\3/' # Format SSN
# 123-45-6789 → (123) 45-6789

# Conditional processing
sed '/pattern/s/old/new/' file.txt   # Replace only in matching lines
sed '/^#/d' file.txt                 # Delete comment lines
```

### 6.3 Multi-line Matching

```bash
#!/bin/bash

# grep multi-line (with -Pzo in GNU grep)
grep -Pzo '(?s)function.*?\{.*?\}' code.js

# sed multi-line
sed -n '/start/,/end/p' file.txt     # Print range

# awk multi-line
awk '/start/,/end/' file.txt
```

## 7. Performance Considerations

### 7.1 Regex Compilation

```bash
#!/bin/bash

# INEFFICIENT: Regex compiled on every iteration
for item in "${items[@]}"; do
    if [[ $item =~ ^[0-9]+$ ]]; then
        process "$item"
    fi
done

# EFFICIENT: Compile once, use many times
pattern='^[0-9]+$'
for item in "${items[@]}"; do
    if [[ $item =~ $pattern ]]; then
        process "$item"
    fi
done
```

### 7.2 Avoiding Catastrophic Backtracking

```bash
#!/bin/bash

# DANGEROUS: Can cause catastrophic backtracking
# Pattern: (a+)+b
# String: "aaaaaaaaaaaaaaaaaaaaaaaac" (no 'b' at end)
# This will take exponential time!

bad_pattern='(a+)+b'
# Avoid nested quantifiers on same content

# SAFE: Use atomic grouping or possessive quantifiers (if supported)
# Or restructure to avoid backtracking
good_pattern='a+b'

# General rule: Avoid patterns like (.*)*,  (.+)+, etc.
```

### 7.3 Simple Operations vs Regex

```bash
#!/bin/bash

# When simple string operations are faster than regex

# Check prefix
# SLOW:
[[ $str =~ ^prefix ]]

# FAST:
[[ $str == prefix* ]]

# Check suffix
# SLOW:
[[ $str =~ suffix$ ]]

# FAST:
[[ $str == *suffix ]]

# Check contains
# SLOW:
[[ $str =~ substring ]]

# FAST:
[[ $str == *substring* ]]

# Use regex when pattern matching is truly needed
# Use simple operations for literal substring checks
```

### 7.4 Benchmarking Patterns

```bash
#!/bin/bash

benchmark_regex() {
    local pattern=$1
    local test_string=$2
    local iterations=10000

    local start=$(date +%s%N)

    for ((i=0; i<iterations; i++)); do
        [[ $test_string =~ $pattern ]] > /dev/null
    done

    local end=$(date +%s%N)
    local elapsed=$(( (end - start) / 1000000 ))

    echo "Pattern: $pattern"
    echo "Time: ${elapsed}ms for $iterations iterations"
    echo "Avg: $((elapsed * 1000 / iterations))μs per match"
}

# Compare patterns
benchmark_regex '^[0-9]+$' "12345"
benchmark_regex '[0-9]' "12345"
```

## 8. Common Patterns Reference

### 8.1 Pattern Library

| Pattern | Description | Example |
|---------|-------------|---------|
| `^[0-9]+$` | Integer | 123, 456 |
| `^[0-9]*\.[0-9]+$` | Decimal | 3.14, 0.5 |
| `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` | Email | user@example.com |
| `^https?://[^\s]+$` | URL | https://example.com |
| `^([0-9]{1,3}\.){3}[0-9]{1,3}$` | IPv4 | 192.168.1.1 |
| `^[0-9]{4}-[0-9]{2}-[0-9]{2}$` | Date YYYY-MM-DD | 2024-02-13 |
| `^([01][0-9]|2[0-3]):[0-5][0-9]$` | Time HH:MM | 14:30 |
| `^[0-9]{3}-[0-9]{2}-[0-9]{4}$` | SSN | 123-45-6789 |
| `^\(\d{3}\) \d{3}-\d{4}$` | Phone (US) | (555) 123-4567 |
| `^/.*$` | Unix path | /path/to/file |
| `^[a-fA-F0-9]{32}$` | MD5 hash | 5d41402abc4b2a76b9719d911017c592 |
| `^[0-9a-f]{40}$` | SHA-1 hash | aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d |
| `^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$` | IBAN | GB82WEST12345698765432 |

### 8.2 Practical Pattern Examples

```bash
#!/bin/bash

# Credit card (simplified - doesn't validate checksum)
CC_PATTERN='^[0-9]{4}[[:space:]-]?[0-9]{4}[[:space:]-]?[0-9]{4}[[:space:]-]?[0-9]{4}$'

# Hex color code
COLOR_PATTERN='^#[0-9a-fA-F]{6}$'

# MAC address
MAC_PATTERN='^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$'

# Username (alphanumeric, underscore, hyphen, 3-16 chars)
USERNAME_PATTERN='^[a-zA-Z0-9_-]{3,16}$'

# Strong password (min 8 chars, uppercase, lowercase, digit, special)
PASSWORD_PATTERN='^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*]).{8,}$'

# Domain name
DOMAIN_PATTERN='^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$'

# File extension
EXT_PATTERN='\.(txt|log|conf|json|yaml|xml)$'
```

## Practice Problems

### Problem 1: Advanced Input Validator
Create a comprehensive input validation library:
- Support multiple validation rules per field (e.g., required, type, length, pattern)
- Implement custom validators (e.g., password strength, credit card, IBAN)
- Return detailed error messages (what failed and why)
- Support conditional validation (field B required if field A has value)
- Validate nested data structures (JSON-like objects)
- Generate validation report with all errors

### Problem 2: Log Parser with Regex
Build a log parser that:
- Auto-detects log format (Apache, Nginx, syslog, custom)
- Extracts timestamp, level, source, message using regex
- Handles multi-line log entries (stack traces, etc.)
- Validates log format and reports malformed entries
- Converts timestamps to different formats
- Filters logs by complex regex patterns
- Generates statistics (top errors, time distribution)

### Problem 3: Data Sanitizer
Implement a data sanitization tool:
- Remove/escape special characters for different contexts (SQL, HTML, shell)
- Validate and normalize phone numbers (multiple formats)
- Validate and normalize email addresses
- Redact sensitive data (SSN, credit cards, API keys) using regex
- Validate and format dates/times
- Handle URLs (parse, validate, normalize)
- Generate sanitization report

### Problem 4: Configuration File Parser
Create a universal config parser:
- Parse INI, YAML-like, and custom key=value formats
- Support sections, nested keys, arrays
- Validate syntax using regex
- Extract values with type conversion (string, int, bool, array)
- Support variable substitution (e.g., ${HOME}/path)
- Validate values against patterns (e.g., port must be 1-65535)
- Support includes and inheritance

### Problem 5: Pattern-Based Router
Build a URL/path router:
- Define routes with patterns (e.g., /user/:id, /posts/:year/:month/:slug)
- Extract path parameters using regex and BASH_REMATCH
- Support optional parameters (/path/:id?, /path/:id*)
- Support regex constraints (/user/:id([0-9]+))
- Match routes by priority
- Generate URLs from patterns and parameters
- Support query string parsing

**Previous**: [07_String_Processing.md](./07_String_Processing.md) | **Next**: [09_Process_Management.md](./09_Process_Management.md)
