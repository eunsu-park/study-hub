# Lesson 05: Functions and Libraries

**Difficulty**: ⭐⭐

**Previous**: [04_Advanced_Control_Flow.md](./04_Advanced_Control_Flow.md) | **Next**: [06_IO_and_Redirection.md](./06_IO_and_Redirection.md)

## 1. Return Value Patterns

Bash functions don't return values like traditional programming languages. Instead, they use several patterns to communicate results back to the caller.

### 1.1 Echo Capture Pattern

The most common pattern is to `echo` the result and capture it with command substitution:

```bash
#!/bin/bash

# Function returns value via echo
add() {
    local sum=$(( $1 + $2 ))
    echo "$sum"
}

# Capture the result
result=$(add 10 20)
echo "Result: $result"  # Output: Result: 30
```

**Advantages**: Clean, functional style; supports multiple return values via multiple echo statements.

**Disadvantages**: Slower due to subshell creation; can't distinguish between stdout and return values.

### 1.2 Global Variable Pattern

Functions can modify global variables directly:

```bash
#!/bin/bash

# Function sets global variable
calculate_stats() {
    local -a numbers=("$@")
    local sum=0
    local count=${#numbers[@]}

    for num in "${numbers[@]}"; do
        ((sum += num))
    done

    # Set global variables
    STATS_SUM=$sum
    STATS_AVG=$(( sum / count ))
    STATS_COUNT=$count
}

calculate_stats 10 20 30 40 50
echo "Sum: $STATS_SUM"        # Output: Sum: 150
echo "Average: $STATS_AVG"    # Output: Average: 30
echo "Count: $STATS_COUNT"    # Output: Count: 5
```

**Advantages**: Fast; can return multiple values easily; no subshell overhead.

**Disadvantages**: Pollutes global namespace; harder to reason about; not thread-safe.

### 1.3 Nameref Pattern (Bash 4.3+)

Use `declare -n` to create a reference to a variable:

```bash
#!/bin/bash

# Function uses nameref to modify caller's variable
get_user_info() {
    local -n result_ref=$1  # Create nameref
    local username=$2

    # Simulate API call
    result_ref=(
        "name=$username"
        "id=12345"
        "email=${username}@example.com"
    )
}

# Call with variable name (not value)
declare -a user_data
get_user_info user_data "john"

for field in "${user_data[@]}"; do
    echo "$field"
done
# Output:
# name=john
# id=12345
# email=john@example.com
```

**Advantages**: Clean separation; no global pollution; can modify caller's variables directly.

**Disadvantages**: Requires Bash 4.3+; slightly complex syntax.

### 1.4 Return Status Code Pattern

Use `return` to set the exit status (0-255):

```bash
#!/bin/bash

# Function returns status code
is_valid_port() {
    local port=$1

    # Validate port number
    if [[ ! $port =~ ^[0-9]+$ ]]; then
        return 1  # Invalid: not a number
    fi

    if (( port < 1 || port > 65535 )); then
        return 2  # Invalid: out of range
    fi

    return 0  # Valid
}

# Test the function
if is_valid_port 8080; then
    echo "Port 8080 is valid"
fi

is_valid_port "abc"
case $? in
    0) echo "Valid port" ;;
    1) echo "Error: Not a number" ;;
    2) echo "Error: Out of range" ;;
esac
# Output: Error: Not a number
```

**Advantages**: Standard Unix convention; good for success/failure checks.

**Disadvantages**: Limited to integers 0-255; often combined with other patterns.

### 1.5 Comparison Table

| Pattern | Speed | Multiple Returns | Complexity | Best Use Case |
|---------|-------|------------------|------------|---------------|
| Echo Capture | Slow | Yes (multiple echoes) | Low | Simple value returns |
| Global Variable | Fast | Yes | Medium | Performance-critical code |
| Nameref | Fast | Yes | Medium | Clean API design |
| Return Status | Fast | No | Low | Success/failure checks |

## 2. Recursive Functions

Recursive functions call themselves to solve problems that have a recursive structure.

### 2.1 Factorial

```bash
#!/bin/bash

# Classic recursive factorial
factorial() {
    local n=$1

    # Base case
    if (( n <= 1 )); then
        echo 1
        return
    fi

    # Recursive case
    local prev=$(factorial $((n - 1)))
    echo $(( n * prev ))
}

echo "5! = $(factorial 5)"  # Output: 5! = 120
```

### 2.2 Directory Tree Traversal

```bash
#!/bin/bash

# Recursively list all files in directory tree
traverse_directory() {
    local dir=$1
    local indent=${2:-""}

    # Process all items in directory
    for item in "$dir"/*; do
        if [[ -d $item ]]; then
            echo "${indent}[DIR]  $(basename "$item")"
            # Recursive call with increased indent
            traverse_directory "$item" "$indent  "
        else
            echo "${indent}[FILE] $(basename "$item")"
        fi
    done
}

# Usage
traverse_directory "/tmp/myproject"
```

### 2.3 Fibonacci with Memoization

Without memoization, recursive Fibonacci is extremely slow. Here's an optimized version:

```bash
#!/bin/bash

# Declare associative array for memoization
declare -A fib_cache

fibonacci() {
    local n=$1

    # Check cache first
    if [[ -n ${fib_cache[$n]} ]]; then
        echo "${fib_cache[$n]}"
        return
    fi

    # Base cases
    if (( n <= 1 )); then
        echo "$n"
        return
    fi

    # Recursive calculation
    local fib1=$(fibonacci $((n - 1)))
    local fib2=$(fibonacci $((n - 2)))
    local result=$((fib1 + fib2))

    # Store in cache
    fib_cache[$n]=$result
    echo "$result"
}

# Calculate Fibonacci numbers
for i in {0..10}; do
    echo "fib($i) = $(fibonacci $i)"
done
```

**Output**:
```
fib(0) = 0
fib(1) = 1
fib(2) = 1
fib(3) = 2
fib(4) = 3
fib(5) = 5
fib(6) = 8
fib(7) = 13
fib(8) = 21
fib(9) = 34
fib(10) = 55
```

## 3. Function Libraries

Organize reusable functions into separate library files for maintainability and reuse.

### 3.1 Creating a Library File

**File: `/opt/mylibs/string_utils.sh`**

```bash
#!/bin/bash
# String utility library

# Convert string to uppercase
str_upper() {
    echo "${1^^}"
}

# Convert string to lowercase
str_lower() {
    echo "${1,,}"
}

# Trim whitespace from both ends
str_trim() {
    local text=$1
    # Remove leading whitespace
    text="${text#"${text%%[![:space:]]*}"}"
    # Remove trailing whitespace
    text="${text%"${text##*[![:space:]]}"}"
    echo "$text"
}

# Check if string contains substring
str_contains() {
    local haystack=$1
    local needle=$2
    [[ $haystack == *"$needle"* ]]
}

# Repeat string N times
str_repeat() {
    local string=$1
    local count=$2
    local result=""

    for ((i=0; i<count; i++)); do
        result+="$string"
    done

    echo "$result"
}
```

### 3.2 Sourcing Library Files

```bash
#!/bin/bash

# Method 1: Absolute path
source /opt/mylibs/string_utils.sh

# Method 2: Relative path
source ./libs/string_utils.sh

# Method 3: Dot notation (equivalent to source)
. /opt/mylibs/string_utils.sh

# Use library functions
text="  Hello World  "
echo "Original: [$text]"
echo "Trimmed: [$(str_trim "$text")]"
echo "Upper: $(str_upper "$text")"
echo "Lower: $(str_lower "$text")"

if str_contains "Hello World" "World"; then
    echo "Contains 'World'"
fi

echo "Repeated: $(str_repeat "=" 40)"
```

### 3.3 Library Path Management

```bash
#!/bin/bash

# Define library path
LIB_PATH="${LIB_PATH:-/opt/mylibs}"

# Function to load a library
load_library() {
    local lib_name=$1
    local lib_file="$LIB_PATH/${lib_name}.sh"

    if [[ ! -f $lib_file ]]; then
        echo "Error: Library '$lib_name' not found at $lib_file" >&2
        return 1
    fi

    source "$lib_file"
}

# Load multiple libraries
load_library "string_utils" || exit 1
load_library "file_utils" || exit 1
load_library "log_utils" || exit 1

# Now use functions from all libraries
```

### 3.4 Library with Initialization

```bash
#!/bin/bash
# Database utilities library

# Library-level variables
declare -g DB_CONNECTION=""
declare -g DB_HOST="localhost"
declare -g DB_PORT=5432

# Library initialization function
db_init() {
    DB_HOST=${1:-$DB_HOST}
    DB_PORT=${2:-$DB_PORT}
    DB_CONNECTION="host=$DB_HOST port=$DB_PORT"
    echo "Database initialized: $DB_CONNECTION"
}

# Database functions
db_query() {
    local query=$1
    if [[ -z $DB_CONNECTION ]]; then
        echo "Error: Database not initialized. Call db_init first." >&2
        return 1
    fi
    # Execute query (simplified)
    echo "Executing: $query on $DB_CONNECTION"
}

# Usage:
# source db_utils.sh
# db_init "dbserver" 3306
# db_query "SELECT * FROM users"
```

## 4. Namespacing

Bash doesn't have built-in namespaces, but we can simulate them with naming conventions.

### 4.1 Prefixed Function Names

```bash
#!/bin/bash

# String utility namespace
stringutil::trim() {
    local text=$1
    text="${text#"${text%%[![:space:]]*}"}"
    text="${text%"${text##*[![:space:]]}"}"
    echo "$text"
}

stringutil::split() {
    local string=$1
    local delimiter=$2
    local -n result_array=$3

    IFS="$delimiter" read -ra result_array <<< "$string"
}

# File utility namespace
fileutil::exists() {
    [[ -f $1 ]]
}

fileutil::size() {
    stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null
}

# Math utility namespace
mathutil::max() {
    local max=$1
    shift
    for num in "$@"; do
        (( num > max )) && max=$num
    done
    echo "$max"
}

mathutil::min() {
    local min=$1
    shift
    for num in "$@"; do
        (( num < min )) && min=$num
    done
    echo "$min"
}

# Usage
text="  Hello  "
echo "Trimmed: [$(stringutil::trim "$text")]"

declare -a parts
stringutil::split "one,two,three" "," parts
echo "Parts: ${parts[@]}"

echo "Max: $(mathutil::max 10 5 20 15)"
echo "Min: $(mathutil::min 10 5 20 15)"
```

### 4.2 Namespace Helper Functions

```bash
#!/bin/bash

# Create namespace-aware function loader
namespace() {
    local ns=$1
    shift

    for func in "$@"; do
        eval "${ns}::${func}() { ${func} \"\$@\"; }"
    done
}

# Define regular functions
add() { echo $(( $1 + $2 )); }
subtract() { echo $(( $1 - $2 )); }
multiply() { echo $(( $1 * $2 )); }

# Create namespaced versions
namespace math add subtract multiply

# Use both versions
echo "Direct: $(add 10 5)"           # Output: 15
echo "Namespaced: $(math::add 10 5)" # Output: 15
```

## 5. Callback Patterns

Pass function names as arguments to create flexible, event-driven code.

### 5.1 Simple Callback

```bash
#!/bin/bash

# Process each item with a callback function
process_array() {
    local -n array=$1
    local callback=$2

    for item in "${array[@]}"; do
        $callback "$item"
    done
}

# Callback functions
print_uppercase() {
    echo "${1^^}"
}

print_with_prefix() {
    echo ">>> $1"
}

# Usage
fruits=("apple" "banana" "cherry")

echo "Uppercase:"
process_array fruits print_uppercase

echo -e "\nWith prefix:"
process_array fruits print_with_prefix
```

### 5.2 Event Handler Pattern

```bash
#!/bin/bash

# Event handler registry
declare -A event_handlers

# Register event handler
on() {
    local event=$1
    local handler=$2
    event_handlers[$event]+="$handler "
}

# Trigger event
trigger() {
    local event=$1
    shift
    local handlers=${event_handlers[$event]}

    if [[ -n $handlers ]]; then
        for handler in $handlers; do
            $handler "$@"
        done
    fi
}

# Define event handlers
on_file_created() {
    echo "[INFO] File created: $1"
}

on_file_validated() {
    echo "[INFO] File validated: $1"
}

log_event() {
    echo "[LOG] $(date): $1" >> events.log
}

# Register handlers
on "file.created" on_file_created
on "file.created" log_event
on "file.validated" on_file_validated

# Trigger events
trigger "file.created" "data.txt"
trigger "file.validated" "data.txt"
```

### 5.3 Filter and Map Pattern

```bash
#!/bin/bash

# Map function: apply callback to each element
map() {
    local -n input_array=$1
    local -n output_array=$2
    local callback=$3

    output_array=()
    for item in "${input_array[@]}"; do
        output_array+=("$($callback "$item")")
    done
}

# Filter function: keep elements where callback returns 0
filter() {
    local -n input_array=$1
    local -n output_array=$2
    local predicate=$3

    output_array=()
    for item in "${input_array[@]}"; do
        if $predicate "$item"; then
            output_array+=("$item")
        fi
    done
}

# Callback functions
double() {
    echo $(( $1 * 2 ))
}

is_even() {
    (( $1 % 2 == 0 ))
}

# Usage
numbers=(1 2 3 4 5 6 7 8 9 10)

declare -a doubled
map numbers doubled double
echo "Doubled: ${doubled[@]}"
# Output: Doubled: 2 4 6 8 10 12 14 16 18 20

declare -a evens
filter numbers evens is_even
echo "Evens: ${evens[@]}"
# Output: Evens: 2 4 6 8 10
```

## 6. Variable Scope

Understanding variable scope is crucial for writing correct functions.

### 6.1 Local vs Global Variables

```bash
#!/bin/bash

# Global variable
global_var="I am global"

test_scope() {
    # Local variable (only visible in this function)
    local local_var="I am local"

    # Modify global variable
    global_var="Modified by function"

    echo "Inside function:"
    echo "  Local: $local_var"
    echo "  Global: $global_var"
}

echo "Before function:"
echo "  Global: $global_var"

test_scope

echo "After function:"
echo "  Global: $global_var"
echo "  Local: $local_var"  # Empty - not accessible here
```

**Output**:
```
Before function:
  Global: I am global
Inside function:
  Local: I am local
  Global: Modified by function
After function:
  Global: Modified by function
  Local:
```

### 6.2 Dynamic Scoping in Bash

Bash uses dynamic scoping, not lexical scoping:

```bash
#!/bin/bash

var="global"

outer() {
    local var="outer"
    inner
}

inner() {
    echo "Inner sees: $var"
}

echo "Direct call:"
inner  # Output: Inner sees: global

echo "Call via outer:"
outer  # Output: Inner sees: outer
```

### 6.3 Local Nameref

```bash
#!/bin/bash

# Modify associative array via nameref
update_config() {
    local -n config=$1
    local key=$2
    local value=$3

    config[$key]=$value
}

# Create config
declare -A app_config=(
    [host]="localhost"
    [port]="8080"
    [debug]="false"
)

echo "Before: ${app_config[port]}"
update_config app_config "port" "9090"
echo "After: ${app_config[port]}"
```

### 6.4 Avoiding Common Pitfalls

```bash
#!/bin/bash

# WRONG: Variable leaks to global scope
wrong_function() {
    result=$(( $1 + $2 ))  # result is global!
}

# RIGHT: Use local
right_function() {
    local result=$(( $1 + $2 ))
    echo "$result"
}

# WRONG: Nameref conflict
wrong_nameref() {
    local -n ref=$1
    local ref="something"  # Error: ref is already a nameref
}

# RIGHT: Use different variable names
right_nameref() {
    local -n ref=$1
    local value="something"
    ref="$value"
}
```

## 7. Function Best Practices

### 7.1 Documentation Comments

```bash
#!/bin/bash

#
# Calculate the greatest common divisor of two numbers
#
# Arguments:
#   $1 - First number (positive integer)
#   $2 - Second number (positive integer)
#
# Returns:
#   Prints the GCD to stdout
#
# Example:
#   gcd 48 18  # Output: 6
#
gcd() {
    local a=$1
    local b=$2

    while (( b != 0 )); do
        local temp=$b
        b=$(( a % b ))
        a=$temp
    done

    echo "$a"
}

#
# Parse command-line arguments into an associative array
#
# Arguments:
#   Variable arguments in --key=value or --key value format
#
# Outputs:
#   Sets global associative array ARGS with parsed values
#
# Example:
#   parse_args --host=localhost --port 8080 --verbose
#   # ARGS[host] = "localhost"
#   # ARGS[port] = "8080"
#   # ARGS[verbose] = "true"
#
parse_args() {
    declare -gA ARGS

    while [[ $# -gt 0 ]]; do
        case $1 in
            --*=*)
                key="${1%%=*}"
                key="${key#--}"
                value="${1#*=}"
                ARGS[$key]="$value"
                shift
                ;;
            --*)
                key="${1#--}"
                if [[ $2 != --* ]] && [[ -n $2 ]]; then
                    ARGS[$key]="$2"
                    shift 2
                else
                    ARGS[$key]="true"
                    shift
                fi
                ;;
            *)
                shift
                ;;
        esac
    done
}
```

### 7.2 Input Validation

```bash
#!/bin/bash

#
# Safe division with comprehensive input validation
#
divide() {
    local numerator=$1
    local denominator=$2

    # Check argument count
    if (( $# != 2 )); then
        echo "Error: divide requires exactly 2 arguments" >&2
        return 1
    fi

    # Validate numerator is a number
    if [[ ! $numerator =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: numerator must be a number" >&2
        return 2
    fi

    # Validate denominator is a number
    if [[ ! $denominator =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: denominator must be a number" >&2
        return 2
    fi

    # Check for division by zero
    if (( $(echo "$denominator == 0" | bc -l) )); then
        echo "Error: division by zero" >&2
        return 3
    fi

    # Perform division
    echo "scale=4; $numerator / $denominator" | bc -l
}

# Usage
divide 10 2      # Output: 5.0000
divide 10 0      # Error: division by zero
divide 10        # Error: divide requires exactly 2 arguments
divide 10 "abc"  # Error: denominator must be a number
```

### 7.3 Error Handling in Functions

```bash
#!/bin/bash

#
# Process file with comprehensive error handling
#
process_file() {
    local file=$1
    local -n result=$2

    # Input validation
    if [[ -z $file ]]; then
        echo "Error: file path required" >&2
        return 1
    fi

    # Check file exists
    if [[ ! -f $file ]]; then
        echo "Error: file not found: $file" >&2
        return 2
    fi

    # Check file is readable
    if [[ ! -r $file ]]; then
        echo "Error: file not readable: $file" >&2
        return 3
    fi

    # Process file (with error handling)
    local line_count
    if ! line_count=$(wc -l < "$file" 2>&1); then
        echo "Error: failed to count lines: $line_count" >&2
        return 4
    fi

    local word_count
    if ! word_count=$(wc -w < "$file" 2>&1); then
        echo "Error: failed to count words: $word_count" >&2
        return 5
    fi

    # Return results via nameref
    result=(
        "file=$file"
        "lines=$line_count"
        "words=$word_count"
    )

    return 0
}

# Usage with error handling
declare -a file_stats

if process_file "data.txt" file_stats; then
    echo "Success:"
    printf '  %s\n' "${file_stats[@]}"
else
    error_code=$?
    echo "Failed with error code: $error_code"
fi
```

### 7.4 Function Template

```bash
#!/bin/bash

#
# Function description
#
# Arguments:
#   $1 - Description of first argument
#   $2 - Description of second argument
#   ...
#
# Environment Variables:
#   VAR_NAME - Description (if applicable)
#
# Returns:
#   0 - Success
#   1 - Error description
#   2 - Another error description
#
# Outputs:
#   Description of what is printed to stdout
#
# Side Effects:
#   Description of any global state changes
#
# Example:
#   function_name arg1 arg2
#
function_name() {
    # Validate arguments
    if (( $# < 2 )); then
        echo "Error: insufficient arguments" >&2
        return 1
    fi

    # Local variables
    local arg1=$1
    local arg2=$2
    local result

    # Input validation
    # ...

    # Main logic
    # ...

    # Return/output results
    echo "$result"
    return 0
}
```

## Practice Problems

### Problem 1: String Utility Library
Create a string utility library (`string_lib.sh`) with the following functions:
- `str_reverse()` - Reverse a string
- `str_is_palindrome()` - Check if string is palindrome (return 0 for true)
- `str_count_words()` - Count words in a string
- `str_capitalize()` - Capitalize first letter of each word
- `str_remove_duplicates()` - Remove duplicate consecutive characters

Test your library with a separate script.

### Problem 2: Recursive File Search
Write a recursive function `find_files()` that:
- Takes a directory path and file pattern (e.g., "*.txt")
- Recursively searches all subdirectories
- Returns array of matching file paths via nameref
- Handles symbolic links safely (avoid infinite loops)
- Counts total files searched and matches found

### Problem 3: Calculator with Callbacks
Create a calculator that:
- Accepts a callback function for each operation (+, -, *, /)
- Supports registering custom operations via `register_op()`
- Validates input and handles errors
- Maintains operation history
- Example: `calc 10 "+" 5` → uses registered addition callback

### Problem 4: Configuration Manager
Build a configuration management system with:
- `config_load()` - Load config from file into associative array
- `config_get()` - Get value with default fallback
- `config_set()` - Set value with validation callback
- `config_save()` - Save config back to file
- `config_watch()` - Reload config if file changes (callback on change)

Use namespacing (config::*) and proper error handling.

### Problem 5: Function Performance Profiler
Write a profiler that:
- Wraps any function to measure execution time
- Tracks call count for each function
- Records min/max/average execution time
- Generates a performance report
- Example: `profile my_function arg1 arg2`

**Previous**: [04_Advanced_Control_Flow.md](./04_Advanced_Control_Flow.md) | **Next**: [06_IO_and_Redirection.md](./06_IO_and_Redirection.md)
