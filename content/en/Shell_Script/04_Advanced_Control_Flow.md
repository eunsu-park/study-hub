# Advanced Control Flow ⭐⭐

**Previous**: [03_Arrays_and_Data.md](./03_Arrays_and_Data.md) | **Next**: [05_Functions_and_Libraries.md](./05_Functions_and_Libraries.md)

---

This lesson covers advanced control flow constructs in bash, including different test command variants, arithmetic evaluation, pattern matching, interactive menus, and sophisticated flow control patterns.

## 1. Test Commands Comparison

Bash provides multiple ways to test conditions. Understanding their differences is crucial for writing correct and efficient scripts.

### The Three Test Constructs

```bash
#!/usr/bin/env bash

# 1. [ ] - POSIX test command (alias for 'test')
# 2. [[ ]] - Bash keyword (extended test)
# 3. (( )) - Arithmetic evaluation

# [ ] - Single bracket (POSIX)
if [ "$USER" = "root" ]; then
    echo "Running as root"
fi

# [[ ]] - Double bracket (Bash)
if [[ "$USER" == "root" ]]; then
    echo "Running as root"
fi

# (( )) - Arithmetic
if (( UID == 0 )); then
    echo "Running as root"
fi
```

### Comprehensive Comparison Table

| Feature | `[ ]` (test) | `[[ ]]` (keyword) | `(( ))` (arithmetic) |
|---------|-------------|-------------------|---------------------|
| **POSIX Compliant** | Yes | No (bash only) | No (bash only) |
| **Word Splitting** | Yes (dangerous) | No (safe) | N/A |
| **Pathname Expansion** | Yes (dangerous) | No (safe) | N/A |
| **Pattern Matching** | No | Yes (`==`, `!=` with globs) | No |
| **Regex Matching** | No | Yes (`=~`) | No |
| **String Comparison** | `=`, `!=` | `=`, `==`, `!=`, `<`, `>` | No |
| **Numeric Comparison** | `-eq`, `-ne`, `-lt`, `-le`, `-gt`, `-ge` | Same | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| **Logical Operators** | `-a`, `-o`, `!` | `&&`, `||`, `!` | `&&`, `||`, `!` |
| **Grouping** | `\( \)` (escaped) | `( )` (normal) | `( )` (normal) |
| **Variable Quoting** | Required | Optional | N/A |
| **Performance** | Slower (external command) | Faster (builtin) | Fastest |

### When to Use Each

```bash
#!/usr/bin/env bash

# Use [ ] for POSIX compatibility
if [ -f /etc/passwd ]; then
    echo "Password file exists"
fi

# Use [[ ]] for bash-specific features
# - Pattern matching
if [[ "$filename" == *.txt ]]; then
    echo "Text file"
fi

# - Regex matching
if [[ "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
    echo "Valid email"
fi

# - Safe with unquoted variables
if [[ $var == "value" ]]; then  # Safe even if $var is empty
    echo "Match"
fi

# Use (( )) for arithmetic
if (( count > 100 )); then
    echo "Count exceeds 100"
fi

# - Complex arithmetic
if (( (x + y) * 2 > threshold )); then
    echo "Calculation exceeds threshold"
fi
```

### Dangerous Patterns with [ ]

```bash
#!/usr/bin/env bash

# DANGEROUS: Word splitting
file="my document.txt"
[ -f $file ]  # Expands to: [ -f my document.txt ]
              # Error: too many arguments

# SAFE: Quote the variable
[ -f "$file" ]  # Correct

# DANGEROUS: Glob expansion
pattern="*.txt"
[ -f $pattern ]  # Expands to: [ -f file1.txt file2.txt ... ]
                 # Error: too many arguments

# SAFE: Quote or use [[
[ -f "$pattern" ]  # Checks for file literally named "*.txt"
[[ -f $pattern ]]  # No expansion, safe even unquoted

# DANGEROUS: Empty variables
var=""
[ $var = "value" ]  # Expands to: [ = "value" ]
                    # Error: unary operator expected

# SAFE: Quote the variable
[ "$var" = "value" ]  # Correct
[[ $var = "value" ]]  # Safe even unquoted
```

## 2. [[ ]] Advanced Features

Double brackets provide powerful features not available in single brackets.

### Pattern Matching

```bash
#!/usr/bin/env bash

filename="document.txt"

# Glob pattern matching with ==
if [[ "$filename" == *.txt ]]; then
    echo "Text file"
fi

# Multiple patterns
if [[ "$filename" == *.txt || "$filename" == *.md ]]; then
    echo "Document file"
fi

# Negation
if [[ "$filename" != *.log ]]; then
    echo "Not a log file"
fi

# Case-insensitive (with nocaseglob)
shopt -s nocaseglob
if [[ "$filename" == *.TXT ]]; then
    echo "Text file (case-insensitive)"
fi
shopt -u nocaseglob

# Complex patterns (requires extglob)
shopt -s extglob
if [[ "$filename" == +(*.txt|*.md|*.rst) ]]; then
    echo "Documentation file"
fi
```

### Regex Matching

```bash
#!/usr/bin/env bash

# =~ operator for regex matching

# Email validation
email="user@example.com"
if [[ "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
    echo "Valid email"
fi

# Extract parts using BASH_REMATCH
if [[ "$email" =~ ^([^@]+)@(.+)$ ]]; then
    username="${BASH_REMATCH[1]}"
    domain="${BASH_REMATCH[2]}"
    echo "Username: $username"
    echo "Domain: $domain"
fi

# IP address validation
ip="192.168.1.1"
if [[ "$ip" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
    echo "Valid IP format"
fi

# Phone number
phone="(555) 123-4567"
if [[ "$phone" =~ ^\([0-9]{3}\)\ [0-9]{3}-[0-9]{4}$ ]]; then
    echo "Valid phone number"
fi

# URL parsing
url="https://example.com:8080/path"
if [[ "$url" =~ ^(https?://)?([^:/]+)(:([0-9]+))?(/.*)?$ ]]; then
    protocol="${BASH_REMATCH[1]}"
    host="${BASH_REMATCH[2]}"
    port="${BASH_REMATCH[4]}"
    path="${BASH_REMATCH[5]}"

    echo "Protocol: ${protocol:-http://}"
    echo "Host: $host"
    echo "Port: ${port:-80}"
    echo "Path: ${path:-/}"
fi
```

### String Comparison

```bash
#!/usr/bin/env bash

# [[ ]] supports < and > for lexicographic comparison

str1="apple"
str2="banana"

if [[ "$str1" < "$str2" ]]; then
    echo "$str1 comes before $str2"
fi

if [[ "$str1" > "$str2" ]]; then
    echo "$str1 comes after $str2"
fi

# Version comparison (simple)
ver1="1.2.3"
ver2="1.10.0"

# This is lexicographic, not numeric!
if [[ "$ver1" < "$ver2" ]]; then
    echo "$ver1 < $ver2"
else
    echo "$ver1 >= $ver2"
fi
# Output: 1.2.3 >= 1.10.0 (wrong! because "2" > "1" lexicographically)
```

### Logical Operators

```bash
#!/usr/bin/env bash

# && (AND), || (OR), ! (NOT)

age=25
citizen=true

# AND
if [[ $age -ge 18 && "$citizen" == "true" ]]; then
    echo "Eligible to vote"
fi

# OR
if [[ $age -lt 13 || $age -gt 65 ]]; then
    echo "Discounted ticket"
fi

# NOT
if [[ ! -f /tmp/lock ]]; then
    echo "No lock file"
fi

# Grouping
if [[ ( $age -lt 18 || ! "$citizen" == "true" ) && -f /tmp/register ]]; then
    echo "Need to register"
fi

# Complex conditions
if [[ "$env" == "prod" && ( "$user" == "admin" || "$user" == "root" ) ]]; then
    echo "Production admin access"
fi
```

### Preventing Word Splitting and Globbing

```bash
#!/usr/bin/env bash

# [[ ]] doesn't perform word splitting or glob expansion

files="*.txt"
sentence="hello world"

# With [ ], this would fail
# [ $sentence = "hello world" ]  # Error: too many arguments

# With [[ ]], it works fine
if [[ $sentence = "hello world" ]]; then
    echo "Match (unquoted variable works)"
fi

# No glob expansion
if [[ $files = "*.txt" ]]; then
    echo "Literal match (no expansion)"
fi

# Still, quoting is good practice for clarity
if [[ "$files" = "*.txt" ]]; then
    echo "Quoted match"
fi
```

## 3. Arithmetic Evaluation

Bash provides several ways to perform arithmetic operations.

### Arithmetic Methods Comparison

```bash
#!/usr/bin/env bash

# Four methods for arithmetic

# 1. $(( )) - Arithmetic expansion (POSIX)
result=$((5 + 3))
echo "5 + 3 = $result"

# 2. let - Builtin command
let result=5+3
echo "5 + 3 = $result"

# 3. (( )) - Arithmetic evaluation (returns exit status)
(( result = 5 + 3 ))
echo "5 + 3 = $result"

# 4. expr - External command (deprecated, slow)
result=$(expr 5 + 3)
echo "5 + 3 = $result"
```

### Comparison of Arithmetic Methods

| Method | POSIX | Speed | Use Case |
|--------|-------|-------|----------|
| `$(( ))` | Yes | Fast | Calculations, assignments |
| `let` | No | Fast | Multiple assignments |
| `(( ))` | No | Fast | Conditionals, loops |
| `expr` | Yes | Slow | Avoid (legacy) |
| `bc` | External | Slow | Floating point |

### $(( )) Arithmetic Expansion

```bash
#!/usr/bin/env bash

# Basic operations
echo $((5 + 3))      # 8
echo $((10 - 4))     # 6
echo $((6 * 7))      # 42
echo $((20 / 3))     # 6 (integer division)
echo $((20 % 3))     # 2 (modulo)
echo $((2 ** 10))    # 1024 (exponentiation)

# Variables (no $ needed inside $(( )))
a=5
b=3
echo $((a + b))      # 8
echo $((a * b))      # 15

# Assignment
result=$((a * b + 2))
echo $result         # 17

# Increment/decrement
count=10
echo $((count++))    # 10 (post-increment)
echo $count          # 11

echo $((++count))    # 12 (pre-increment)
echo $count          # 12

echo $((count--))    # 12 (post-decrement)
echo $count          # 11

echo $((--count))    # 10 (pre-decrement)
echo $count          # 10

# Compound assignment
count=5
echo $((count += 3)) # 8
echo $((count *= 2)) # 16
echo $((count /= 4)) # 4
echo $((count %= 3)) # 1

# Bitwise operations
echo $((8 & 4))      # 0 (AND)
echo $((8 | 4))      # 12 (OR)
echo $((8 ^ 4))      # 12 (XOR)
echo $((~8))         # -9 (NOT)
echo $((8 << 2))     # 32 (left shift)
echo $((8 >> 2))     # 2 (right shift)

# Ternary operator
age=20
status=$((age >= 18 ? 1 : 0))
echo $status         # 1
```

### let Command

```bash
#!/usr/bin/env bash

# Multiple assignments
let a=5 b=10 c=15

# Arithmetic
let "result = a + b * c"
echo $result  # 155

# No spaces around = when unquoted
let result=a+b
echo $result  # 15

# Increment
let count=10
let count++
echo $count  # 11

# Multiple operations
let "x = 5" "y = 10" "z = x + y"
echo $z  # 15
```

### (( )) Arithmetic Evaluation

```bash
#!/usr/bin/env bash

# Used in conditionals (returns exit status)
count=5

if (( count > 0 )); then
    echo "Count is positive"
fi

if (( count >= 5 && count <= 10 )); then
    echo "Count in range"
fi

# Used in loops
for (( i=0; i<5; i++ )); do
    echo "Iteration $i"
done

# Assignment
(( result = 5 + 3 ))
echo $result  # 8

# Multiple operations
(( a = 5, b = 10, c = a + b ))
echo $c  # 15

# As standalone command
(( count++ ))
echo $count  # 6
```

### Integer Limitations and Overflow

```bash
#!/usr/bin/env bash

# Bash uses signed long integers (typically 64-bit)

# Maximum value (2^63 - 1)
max=$((2**63 - 1))
echo "Max: $max"
# Max: 9223372036854775807

# Overflow wraps around
overflowed=$((max + 1))
echo "Overflowed: $overflowed"
# Overflowed: -9223372036854775808

# No floating point
result=$((10 / 3))
echo $result  # 3 (not 3.333...)

# Division by zero
# echo $((5 / 0))  # Error: division by 0
```

## 4. Floating Point with bc

For floating-point arithmetic, use the `bc` calculator.

### Basic bc Usage

```bash
#!/usr/bin/env bash

# Simple calculation
result=$(echo "10 / 3" | bc -l)
echo $result  # 3.33333333333333333333

# Set precision (scale)
result=$(echo "scale=2; 10 / 3" | bc)
echo $result  # 3.33

# Multiple operations
result=$(echo "scale=4; (10 + 5) / 3" | bc)
echo $result  # 5.0000

# Using variables
a=10
b=3
result=$(echo "scale=2; $a / $b" | bc)
echo $result  # 3.33
```

### Here-Document with bc

```bash
#!/usr/bin/env bash

# Multi-line bc script
result=$(bc -l <<EOF
    scale=2
    a = 10
    b = 3
    c = a / b
    c * 100
EOF
)
echo $result  # 333.33

# Complex calculation
calculate_compound_interest() {
    local principal=$1
    local rate=$2
    local time=$3
    local n=$4  # compounds per year

    bc -l <<EOF
        scale=2
        p = $principal
        r = $rate / 100
        t = $time
        n = $n
        a = p * (1 + r/n)^(n*t)
        print a
EOF
}

amount=$(calculate_compound_interest 1000 5 10 12)
echo "Amount: \$$amount"
```

### Comparison Idiom

```bash
#!/usr/bin/env bash

# bc returns 1 for true, 0 for false

compare_float() {
    local a=$1
    local op=$2
    local b=$3

    result=$(echo "$a $op $b" | bc -l)
    [ "$result" -eq 1 ]
}

# Usage
if compare_float 3.14 ">" 3.0; then
    echo "3.14 > 3.0"
fi

if compare_float 2.5 "<=" 2.5; then
    echo "2.5 <= 2.5"
fi

# Inline
if (( $(echo "3.14 > 3.0" | bc -l) )); then
    echo "3.14 > 3.0"
fi
```

### Practical Calculations

```bash
#!/usr/bin/env bash

# Percentage calculation
calculate_percentage() {
    local value=$1
    local total=$2
    echo "scale=2; ($value / $total) * 100" | bc
}

score=$(calculate_percentage 45 50)
echo "Score: ${score}%"

# Average
calculate_average() {
    local sum=0
    local count=$#

    for num in "$@"; do
        sum=$(echo "$sum + $num" | bc)
    done

    echo "scale=2; $sum / $count" | bc
}

avg=$(calculate_average 85.5 90.0 78.5 92.0)
echo "Average: $avg"

# Temperature conversion
celsius_to_fahrenheit() {
    local celsius=$1
    echo "scale=2; ($celsius * 9/5) + 32" | bc
}

fahrenheit_to_celsius() {
    local fahrenheit=$1
    echo "scale=2; ($fahrenheit - 32) * 5/9" | bc
}

temp_f=$(celsius_to_fahrenheit 25)
echo "25°C = ${temp_f}°F"

temp_c=$(fahrenheit_to_celsius 77)
echo "77°F = ${temp_c}°C"

# Unit conversion
miles_to_km() {
    echo "scale=2; $1 * 1.60934" | bc
}

km_to_miles() {
    echo "scale=2; $1 / 1.60934" | bc
}

echo "10 miles = $(miles_to_km 10) km"
echo "10 km = $(km_to_miles 10) miles"
```

## 5. Extended Globbing (extglob)

Extended globbing provides powerful pattern matching capabilities.

### Enabling extglob

```bash
#!/usr/bin/env bash

# Enable extended globbing
shopt -s extglob

# Check if enabled
shopt -q extglob && echo "extglob is enabled"
```

### Extended Glob Patterns

```bash
#!/usr/bin/env bash
shopt -s extglob

# ?(pattern) - Matches zero or one occurrence
# *(pattern) - Matches zero or more occurrences
# +(pattern) - Matches one or more occurrences
# @(pattern) - Matches exactly one occurrence
# !(pattern) - Matches anything except pattern

# Create test files
touch file.txt file.log file.bak test.txt data.csv

# ?(pattern) - zero or one
shopt -s nullglob
echo file.?(txt|log)
# file.txt file.log

# *(pattern) - zero or more
# Match files with any number of .bak extensions
touch file.bak file.bak.bak
echo *.*(bak)
# Lists all files

# +(pattern) - one or more
# Match files ending with one or more digits
touch file1.txt file22.txt file333.txt
echo file+([0-9]).txt
# file1.txt file22.txt file333.txt

# @(pattern) - exactly one
echo file.@(txt|log|csv)
# file.txt file.log

# !(pattern) - negation
echo !(*.txt)
# Lists all files except .txt files

# Clean up
rm -f file.* test.txt data.csv
```

### File Matching Examples

```bash
#!/usr/bin/env bash
shopt -s extglob

# Remove all except specific types
# rm !(*.txt|*.log)  # Remove all except .txt and .log

# Match version numbers
# file-1.2.3.tar.gz
# file-+([0-9]).+([0-9]).+([0-9]).tar.gz

# Match optional prefix
# ?(pre-)test.txt matches both "test.txt" and "pre-test.txt"

# Match multiple extensions
# backup.@(tar.gz|zip|tar.bz2)
```

### Using in case Statements

```bash
#!/usr/bin/env bash
shopt -s extglob

filename="$1"

case "$filename" in
    *.@(jpg|jpeg|png|gif))
        echo "Image file"
        ;;
    *.@(mp4|avi|mkv|mov))
        echo "Video file"
        ;;
    *.@(mp3|wav|flac|ogg))
        echo "Audio file"
        ;;
    *.@(txt|md|rst|doc))
        echo "Document file"
        ;;
    *.+(tar.@(gz|bz2|xz)|zip|rar))
        echo "Archive file"
        ;;
    !(*.))
        echo "No extension"
        ;;
    *)
        echo "Unknown file type"
        ;;
esac
```

## 6. select Menu

The `select` command creates interactive menus easily.

### Basic select Menu

```bash
#!/usr/bin/env bash

# Simple menu
options=("Install" "Update" "Remove" "Quit")

select choice in "${options[@]}"; do
    case $choice in
        Install)
            echo "Installing..."
            break
            ;;
        Update)
            echo "Updating..."
            break
            ;;
        Remove)
            echo "Removing..."
            break
            ;;
        Quit)
            echo "Goodbye!"
            break
            ;;
        *)
            echo "Invalid option $REPLY"
            ;;
    esac
done
```

### Custom PS3 Prompt

```bash
#!/usr/bin/env bash

# Customize the select prompt
PS3="Please select an option: "

options=("Start Server" "Stop Server" "Restart Server" "Exit")

select choice in "${options[@]}"; do
    case $choice in
        "Start Server")
            echo "Starting server..."
            ;;
        "Stop Server")
            echo "Stopping server..."
            ;;
        "Restart Server")
            echo "Restarting server..."
            ;;
        "Exit")
            echo "Exiting..."
            break
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done
```

### Input Validation

```bash
#!/usr/bin/env bash

PS3="Select environment: "

environments=("Development" "Staging" "Production")

select env in "${environments[@]}"; do
    # Check if choice is valid
    if [ -n "$env" ]; then
        echo "You selected: $env"

        # Confirm for production
        if [ "$env" = "Production" ]; then
            read -p "Are you sure you want to deploy to production? (yes/no): " confirm
            if [ "$confirm" = "yes" ]; then
                echo "Deploying to production..."
                break
            else
                echo "Deployment cancelled."
            fi
        else
            echo "Deploying to $env..."
            break
        fi
    else
        echo "Invalid selection. Please try again."
    fi
done
```

### Nested Menus

```bash
#!/usr/bin/env bash

main_menu() {
    PS3="Main Menu: "
    options=("Database" "Web Server" "Cache" "Exit")

    select choice in "${options[@]}"; do
        case $choice in
            Database)
                database_menu
                ;;
            "Web Server")
                webserver_menu
                ;;
            Cache)
                cache_menu
                ;;
            Exit)
                echo "Goodbye!"
                return
                ;;
            *)
                echo "Invalid option"
                ;;
        esac
    done
}

database_menu() {
    PS3="Database Menu: "
    options=("Start" "Stop" "Backup" "Back")

    select choice in "${options[@]}"; do
        case $choice in
            Start)
                echo "Starting database..."
                ;;
            Stop)
                echo "Stopping database..."
                ;;
            Backup)
                echo "Backing up database..."
                ;;
            Back)
                return
                ;;
            *)
                echo "Invalid option"
                ;;
        esac
    done
}

webserver_menu() {
    PS3="Web Server Menu: "
    options=("Start" "Stop" "Reload" "Back")

    select choice in "${options[@]}"; do
        case $choice in
            Start|Stop|Reload)
                echo "${choice}ing web server..."
                ;;
            Back)
                return
                ;;
            *)
                echo "Invalid option"
                ;;
        esac
    done
}

cache_menu() {
    PS3="Cache Menu: "
    options=("Start" "Stop" "Clear" "Back")

    select choice in "${options[@]}"; do
        case $choice in
            Start|Stop)
                echo "${choice}ing cache..."
                ;;
            Clear)
                echo "Clearing cache..."
                ;;
            Back)
                return
                ;;
            *)
                echo "Invalid option"
                ;;
        esac
    done
}

# Start application
main_menu
```

### Combining select with case

```bash
#!/usr/bin/env bash

# File operations menu
PS3="Select operation: "

files=(*.txt)
files+=("All files" "Quit")

select file in "${files[@]}"; do
    case $REPLY in
        $((${#files[@]}-1)))  # "All files" index
            echo "Processing all files..."
            for f in *.txt; do
                echo "  Processing $f"
            done
            ;;
        ${#files[@]})  # "Quit" index
            echo "Exiting..."
            break
            ;;
        *)
            if [ -n "$file" ] && [ "$file" != "All files" ] && [ "$file" != "Quit" ]; then
                echo "Processing: $file"
                # Process the file
            else
                echo "Invalid selection"
            fi
            ;;
    esac
done
```

## 7. Advanced case Patterns

The `case` statement supports sophisticated pattern matching.

### Multiple Patterns

```bash
#!/usr/bin/env bash

response="$1"

case "$response" in
    y|Y|yes|Yes|YES)
        echo "Affirmative"
        ;;
    n|N|no|No|NO)
        echo "Negative"
        ;;
    q|Q|quit|Quit|QUIT|exit|Exit|EXIT)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Unknown response"
        ;;
esac
```

### Character Ranges

```bash
#!/usr/bin/env bash

char="$1"

case "$char" in
    [a-z])
        echo "Lowercase letter"
        ;;
    [A-Z])
        echo "Uppercase letter"
        ;;
    [0-9])
        echo "Digit"
        ;;
    [[:space:]])
        echo "Whitespace"
        ;;
    [[:punct:]])
        echo "Punctuation"
        ;;
    *)
        echo "Other character"
        ;;
esac
```

### Glob Patterns

```bash
#!/usr/bin/env bash

filename="$1"

case "$filename" in
    *.txt)
        echo "Text file"
        ;;
    *.log)
        echo "Log file"
        ;;
    *.tar.gz|*.tgz)
        echo "Gzipped tarball"
        ;;
    *.tar.bz2|*.tbz2)
        echo "Bzipped tarball"
        ;;
    backup_*)
        echo "Backup file"
        ;;
    *_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].*)
        echo "File with date stamp"
        ;;
    *)
        echo "Unknown file type"
        ;;
esac
```

### Extended Glob in case (requires extglob)

```bash
#!/usr/bin/env bash
shopt -s extglob

file="$1"

case "$file" in
    *.@(jpg|jpeg|png|gif|bmp))
        echo "Image file: $file"
        file_type="image"
        ;;
    *.@(mp4|avi|mkv|mov|wmv))
        echo "Video file: $file"
        file_type="video"
        ;;
    *.@(mp3|wav|flac|ogg|m4a))
        echo "Audio file: $file"
        file_type="audio"
        ;;
    *.@(zip|tar.@(gz|bz2|xz)|rar|7z))
        echo "Archive file: $file"
        file_type="archive"
        ;;
    !(*.))
        echo "No extension: $file"
        file_type="no_extension"
        ;;
    *)
        echo "Unknown file type: $file"
        file_type="unknown"
        ;;
esac

echo "Type: $file_type"
```

### Fall-Through (;& and ;;&)

```bash
#!/usr/bin/env bash

# ;& - Fall through to next pattern (bash 4.0+)
# ;;& - Continue testing patterns (bash 4.0+)

value="$1"

echo "Using ;& (fall through):"
case "$value" in
    [0-9])
        echo "It's a digit"
        ;&
    [a-z])
        echo "It's lowercase (if it was)"
        ;&
    [A-Z])
        echo "It's uppercase (if it was)"
        ;;
esac

echo -e "\nUsing ;;& (continue matching):"
case "$value" in
    [0-9]*)
        echo "Starts with digit"
        ;;&
    *[0-9])
        echo "Ends with digit"
        ;;&
    *[a-z]*)
        echo "Contains lowercase"
        ;;&
    *)
        echo "Matched catchall"
        ;;
esac
```

### HTTP Status Code Example

```bash
#!/usr/bin/env bash

http_code="$1"

case "$http_code" in
    2[0-9][0-9])
        echo "Success"
        severity="info"
        ;;
    3[0-9][0-9])
        echo "Redirection"
        severity="info"
        ;;
    4[0-9][0-9])
        echo "Client Error"
        severity="warning"

        case "$http_code" in
            401)
                echo "  Unauthorized - check credentials"
                ;;
            403)
                echo "  Forbidden - insufficient permissions"
                ;;
            404)
                echo "  Not Found - check URL"
                ;;
        esac
        ;;
    5[0-9][0-9])
        echo "Server Error"
        severity="error"
        ;;
    *)
        echo "Unknown status code"
        severity="unknown"
        ;;
esac

echo "Severity: $severity"
```

## 8. Flow Control Patterns

### Retry Loop with Backoff

```bash
#!/usr/bin/env bash

retry_with_backoff() {
    local max_attempts=$1
    shift
    local command=("$@")

    local attempt=1
    local delay=1

    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt of $max_attempts..."

        if "${command[@]}"; then
            echo "Success!"
            return 0
        fi

        if [ $attempt -lt $max_attempts ]; then
            echo "Failed. Retrying in ${delay}s..."
            sleep $delay
            delay=$((delay * 2))  # Exponential backoff
        fi

        ((attempt++))
    done

    echo "All attempts failed."
    return 1
}

# Usage
unreliable_command() {
    # Simulates command that fails randomly
    if (( RANDOM % 3 == 0 )); then
        return 0
    else
        return 1
    fi
}

retry_with_backoff 5 unreliable_command
```

### State Machine

```bash
#!/usr/bin/env bash

# Simple state machine for order processing
process_order() {
    local state="pending"

    while true; do
        echo "Current state: $state"

        case "$state" in
            pending)
                echo "Validating order..."
                # Validation logic
                if validate_order; then
                    state="validated"
                else
                    state="rejected"
                fi
                ;;
            validated)
                echo "Processing payment..."
                if process_payment; then
                    state="paid"
                else
                    state="payment_failed"
                fi
                ;;
            paid)
                echo "Preparing shipment..."
                if prepare_shipment; then
                    state="shipped"
                else
                    state="shipment_failed"
                fi
                ;;
            shipped)
                echo "Order completed successfully!"
                state="completed"
                ;;
            completed|rejected|payment_failed|shipment_failed)
                echo "Terminal state: $state"
                break
                ;;
            *)
                echo "Unknown state: $state"
                break
                ;;
        esac

        sleep 1  # Simulate processing time
    done
}

validate_order() { return 0; }
process_payment() { return 0; }
prepare_shipment() { return 0; }

process_order
```

### Dispatch Table

```bash
#!/usr/bin/env bash

# Dispatch table using associative array
declare -A commands

# Register commands
commands[start]="start_service"
commands[stop]="stop_service"
commands[restart]="restart_service"
commands[status]="check_status"
commands[reload]="reload_config"

# Command implementations
start_service() { echo "Starting service..."; }
stop_service() { echo "Stopping service..."; }
restart_service() { stop_service; start_service; }
check_status() { echo "Service is running"; }
reload_config() { echo "Reloading configuration..."; }

# Dispatcher
dispatch() {
    local command="$1"

    if [[ -v commands[$command] ]]; then
        ${commands[$command]}
    else
        echo "Unknown command: $command" >&2
        echo "Available commands: ${!commands[*]}" >&2
        return 1
    fi
}

# Usage
dispatch start
dispatch status
dispatch restart
dispatch invalid
```

### Circuit Breaker Pattern

```bash
#!/usr/bin/env bash

# Circuit breaker for service calls
declare -A circuit_breaker=(
    [state]="closed"
    [failures]=0
    [threshold]=3
    [timeout]=10
)

call_service() {
    local service_name="$1"

    case "${circuit_breaker[state]}" in
        closed)
            if make_service_call "$service_name"; then
                circuit_breaker[failures]=0
                return 0
            else
                ((circuit_breaker[failures]++))

                if [ ${circuit_breaker[failures]} -ge ${circuit_breaker[threshold]} ]; then
                    circuit_breaker[state]="open"
                    circuit_breaker[opened_at]=$(date +%s)
                    echo "Circuit breaker OPEN" >&2
                fi
                return 1
            fi
            ;;
        open)
            local now=$(date +%s)
            local elapsed=$((now - circuit_breaker[opened_at]))

            if [ $elapsed -ge ${circuit_breaker[timeout]} ]; then
                circuit_breaker[state]="half_open"
                echo "Circuit breaker HALF-OPEN (testing)" >&2
                call_service "$service_name"
            else
                echo "Circuit breaker OPEN, failing fast" >&2
                return 1
            fi
            ;;
        half_open)
            if make_service_call "$service_name"; then
                circuit_breaker[state]="closed"
                circuit_breaker[failures]=0
                echo "Circuit breaker CLOSED (recovered)" >&2
                return 0
            else
                circuit_breaker[state]="open"
                circuit_breaker[opened_at]=$(date +%s)
                echo "Circuit breaker OPEN (still failing)" >&2
                return 1
            fi
            ;;
    esac
}

make_service_call() {
    local service="$1"
    # Simulate service call
    echo "Calling $service..."
    (( RANDOM % 2 == 0 ))  # 50% success rate
}

# Test circuit breaker
for i in {1..10}; do
    echo "--- Attempt $i ---"
    if call_service "external-api"; then
        echo "Success"
    else
        echo "Failed"
    fi
    sleep 1
done
```

## Practice Problems

### Problem 1: Advanced Input Validator

Create a comprehensive input validation library that:
- Validates different data types (email, URL, IP, phone, credit card, date, etc.)
- Supports custom regex patterns
- Provides detailed error messages
- Returns structured validation results (pass/fail + error details)
- Supports composite validation (multiple validators on one input)
- Performance-optimized (uses appropriate test constructs)

**Example**:
```bash
validate "user@example.com" email
validate "192.168.1.1" ipv4
validate "2024-02-13" date --format "YYYY-MM-DD"
validate "password123" password --min-length 8 --require-digit --require-special
```

### Problem 2: Expression Evaluator

Build an expression evaluator that:
- Parses and evaluates arithmetic expressions
- Supports variables and functions
- Handles floating-point math (via bc integration)
- Supports logical expressions
- Provides error reporting with position
- Includes common math functions (sin, cos, sqrt, etc.)

**Example**:
```bash
eval_expr "2 + 3 * 4"                    # 14
eval_expr "x = 5; y = 10; x + y"         # 15
eval_expr "sqrt(16) + pow(2, 3)"         # 12
eval_expr "if(x > 5, x * 2, x / 2)" x=10 # 20
```

### Problem 3: Pattern Matcher

Develop a pattern matching tool that:
- Matches files against multiple pattern types (glob, regex, extended glob)
- Supports inclusion and exclusion patterns
- Provides pattern explanations
- Optimizes pattern matching order
- Generates pattern from examples
- Validates patterns before use

**Example**:
```bash
pattern_match "file.txt" --glob "*.txt" --exclude "temp_*"
pattern_match "192.168.1.1" --regex "^([0-9]{1,3}\.){3}[0-9]{1,3}$"
pattern_explain "*.@(jpg|png|gif)"
pattern_generate --from-examples "test1.log" "test2.log" "prod.log"
```

### Problem 4: Interactive Wizard

Create an interactive wizard framework that:
- Builds multi-step forms
- Validates input at each step
- Supports conditional steps (skip based on previous answers)
- Provides summary and confirmation
- Allows going back to previous steps
- Saves progress (resume later)
- Supports different input types (text, select, multi-select, yes/no)

**Example**:
```bash
wizard_start "Setup Database"
wizard_step "host" --prompt "Database host" --default "localhost" --validate ipv4_or_hostname
wizard_step "port" --prompt "Port" --default 5432 --validate port_number
wizard_step "ssl" --prompt "Use SSL?" --type yesno
wizard_step "cert" --prompt "Certificate path" --if "ssl==yes" --validate file_exists
wizard_confirm
wizard_execute
```

### Problem 5: Smart Retry System

Implement an advanced retry system with:
- Multiple retry strategies (constant, linear, exponential backoff, fibonacci)
- Jitter (randomness to prevent thundering herd)
- Circuit breaker integration
- Success/failure callbacks
- Timeout per attempt
- Conditional retry (retry only on specific errors)
- Metrics collection (attempts, success rate, average latency)

**Example**:
```bash
retry --strategy exponential --max-attempts 5 --initial-delay 1 --max-delay 30 \
      --jitter 0.1 --timeout 10 --on-success log_success --on-failure alert \
      --retry-on "1,2,3" -- curl https://api.example.com/endpoint
```

---

**Previous**: [03_Arrays_and_Data.md](./03_Arrays_and_Data.md) | **Next**: [05_Functions_and_Libraries.md](./05_Functions_and_Libraries.md)
