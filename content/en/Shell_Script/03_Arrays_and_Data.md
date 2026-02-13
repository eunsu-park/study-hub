# Arrays and Data Structures ⭐⭐

**Previous**: [02_Parameter_Expansion.md](./02_Parameter_Expansion.md) | **Next**: [04_Advanced_Control_Flow.md](./04_Advanced_Control_Flow.md)

---

This lesson covers bash arrays and how to work with structured data. We'll explore indexed and associative arrays, common data structure patterns, parsing CSV and config files, and practical techniques for managing complex data in shell scripts.

## 1. Indexed Array Operations

Indexed arrays store elements with numeric indices starting at 0.

### Array Declaration and Initialization

```bash
#!/usr/bin/env bash

# Empty array
declare -a empty_array

# Array with initial values
fruits=("apple" "banana" "cherry")

# Explicit declaration
declare -a numbers=(1 2 3 4 5)

# Sparse array (indices don't need to be continuous)
sparse[0]="first"
sparse[5]="sixth"
sparse[10]="eleventh"

# Multi-line declaration
servers=(
    "web1.example.com"
    "web2.example.com"
    "web3.example.com"
)

# From command output
files=(*.txt)  # All .txt files in current directory
lines=($(cat file.txt))  # WARNING: splits on whitespace

# Safe way to read lines into array
mapfile -t lines < file.txt
# or
readarray -t lines < file.txt
```

### Accessing Array Elements

```bash
#!/usr/bin/env bash

fruits=("apple" "banana" "cherry" "date")

# Access single element
echo "${fruits[0]}"     # apple
echo "${fruits[2]}"     # cherry

# Access last element
echo "${fruits[-1]}"    # date (bash 4.3+)
echo "${fruits[@]: -1}" # date (older bash)

# Access all elements
echo "${fruits[@]}"     # apple banana cherry date
echo "${fruits[*]}"     # apple banana cherry date

# Difference between @ and *
for fruit in "${fruits[@]}"; do
    echo "Fruit: $fruit"
done
# Iterates 4 times, one per element

for fruit in "${fruits[*]}"; do
    echo "Fruit: $fruit"
done
# Iterates 1 time, all elements as single string

# Array length
echo "${#fruits[@]}"    # 4 (number of elements)
echo "${#fruits[*]}"    # 4 (same)

# Length of specific element
echo "${#fruits[1]}"    # 6 (length of "banana")
```

### Modifying Arrays

```bash
#!/usr/bin/env bash

# Append elements
fruits=("apple" "banana")
fruits+=("cherry")              # Add one
fruits+=("date" "elderberry")   # Add multiple
echo "${fruits[@]}"
# apple banana cherry date elderberry

# Modify element
fruits[1]="blueberry"
echo "${fruits[@]}"
# apple blueberry cherry date elderberry

# Delete element (creates sparse array)
unset 'fruits[2]'
echo "${fruits[@]}"           # apple blueberry date elderberry
echo "${#fruits[@]}"          # 4 (still 4 elements)
echo "${!fruits[@]}"          # 0 1 3 4 (indices)

# Delete entire array
unset fruits

# Prepend elements (create new array)
nums=(1 2 3)
nums=(0 "${nums[@]}")
echo "${nums[@]}"  # 0 1 2 3

# Insert in middle (requires recreation)
arr=(a b c d)
arr=("${arr[@]:0:2}" "X" "${arr[@]:2}")
echo "${arr[@]}"  # a b X c d
```

### Array Slicing

```bash
#!/usr/bin/env bash

# ${array[@]:start:length}

numbers=(0 1 2 3 4 5 6 7 8 9)

# Extract from index 2, length 3
echo "${numbers[@]:2:3}"  # 2 3 4

# Extract from index 5 to end
echo "${numbers[@]:5}"    # 5 6 7 8 9

# Extract last 3 elements
echo "${numbers[@]: -3}"  # 7 8 9

# Copy array
copy=("${numbers[@]}")
echo "${copy[@]}"

# Copy slice
subset=("${numbers[@]:2:5}")
echo "${subset[@]}"  # 2 3 4 5 6
```

### Iteration

```bash
#!/usr/bin/env bash

servers=("web1" "web2" "db1" "cache1")

# Iterate over values
for server in "${servers[@]}"; do
    echo "Processing: $server"
done

# Iterate over indices
for i in "${!servers[@]}"; do
    echo "Server $i: ${servers[$i]}"
done

# Iterate with index using C-style for
for ((i=0; i<${#servers[@]}; i++)); do
    echo "[$i] = ${servers[$i]}"
done

# Iterate in reverse
for ((i=${#servers[@]}-1; i>=0; i--)); do
    echo "${servers[$i]}"
done
```

### Array Copying and Merging

```bash
#!/usr/bin/env bash

# Copy array
original=(1 2 3)
copy=("${original[@]}")
copy[0]=99
echo "Original: ${original[@]}"  # 1 2 3
echo "Copy: ${copy[@]}"          # 99 2 3

# Merge arrays
arr1=(a b c)
arr2=(d e f)
merged=("${arr1[@]}" "${arr2[@]}")
echo "${merged[@]}"  # a b c d e f

# Merge multiple arrays
arr3=(g h i)
all=("${arr1[@]}" "${arr2[@]}" "${arr3[@]}")
echo "${all[@]}"  # a b c d e f g h i

# Append array to array
arr1+=(x y z)
echo "${arr1[@]}"  # a b c x y z
```

## 2. Associative Arrays

Associative arrays (hash maps, dictionaries) use strings as keys instead of numeric indices. Available in bash 4.0+.

### Declaration and Basic Operations

```bash
#!/usr/bin/env bash

# Must declare as associative array
declare -A config

# Assign values
config[host]="localhost"
config[port]=8080
config[database]="myapp"

# Alternative: initialize at declaration
declare -A user=(
    [name]="John Doe"
    [email]="john@example.com"
    [role]="admin"
)

# Access values
echo "${config[host]}"     # localhost
echo "${user[name]}"       # John Doe

# Check if key exists
if [[ -v config[host] ]]; then
    echo "Host is configured"
fi

# Alternative key existence check
if [[ "${config[host]+exists}" == "exists" ]]; then
    echo "Host key exists"
fi

# Get all keys
echo "${!config[@]}"       # host port database

# Get all values
echo "${config[@]}"        # localhost 8080 myapp

# Number of entries
echo "${#config[@]}"       # 3
```

### Iteration

```bash
#!/usr/bin/env bash

declare -A settings=(
    [theme]="dark"
    [language]="en"
    [notifications]="enabled"
    [auto_save]="true"
)

# Iterate over keys
for key in "${!settings[@]}"; do
    echo "$key = ${settings[$key]}"
done

# Iterate over values
for value in "${settings[@]}"; do
    echo "Value: $value"
done

# Sort keys before iteration
for key in $(echo "${!settings[@]}" | tr ' ' '\n' | sort); do
    echo "$key = ${settings[$key]}"
done
```

### Comparison: Indexed vs Associative Arrays

| Feature | Indexed Array | Associative Array |
|---------|---------------|-------------------|
| Declaration | `arr=()` or `declare -a arr` | `declare -A arr` |
| Keys | Integers (0, 1, 2, ...) | Strings |
| Ordering | Preserved | Unordered |
| Available since | All bash versions | Bash 4.0+ |
| Use case | Sequential data | Key-value pairs |
| Iteration | Order guaranteed | Order undefined |
| Access | `${arr[0]}` | `${arr[key]}` |
| All keys | `${!arr[@]}` | `${!arr[@]}` |

### Practical Examples

```bash
#!/usr/bin/env bash

# HTTP status codes
declare -A http_status=(
    [200]="OK"
    [201]="Created"
    [301]="Moved Permanently"
    [400]="Bad Request"
    [401]="Unauthorized"
    [403]="Forbidden"
    [404]="Not Found"
    [500]="Internal Server Error"
)

lookup_status() {
    local code="$1"
    if [[ -v http_status[$code] ]]; then
        echo "$code ${http_status[$code]}"
    else
        echo "$code Unknown"
    fi
}

lookup_status 200  # 200 OK
lookup_status 404  # 404 Not Found
lookup_status 999  # 999 Unknown

# Environment-specific configuration
declare -A db_config_dev=(
    [host]="localhost"
    [port]=5432
    [name]="dev_db"
    [user]="dev_user"
)

declare -A db_config_prod=(
    [host]="db.example.com"
    [port]=5432
    [name]="prod_db"
    [user]="app_user"
)

get_db_config() {
    local env="$1"
    local -n config_ref="db_config_$env"

    echo "Database configuration for $env:"
    for key in "${!config_ref[@]}"; do
        echo "  $key: ${config_ref[$key]}"
    done
}

get_db_config "dev"
get_db_config "prod"
```

### Nested Data Simulation

```bash
#!/usr/bin/env bash

# Simulate nested structure using naming convention
declare -A data

# user.name
data[user.name]="John Doe"
data[user.email]="john@example.com"
data[user.age]=30

# user.address.city
data[user.address.city]="New York"
data[user.address.country]="USA"

# settings.theme
data[settings.theme]="dark"
data[settings.language]="en"

# Access nested data
echo "Name: ${data[user.name]}"
echo "City: ${data[user.address.city]}"
echo "Theme: ${data[settings.theme]}"

# Get all user.address.* keys
for key in "${!data[@]}"; do
    if [[ "$key" == user.address.* ]]; then
        echo "$key = ${data[$key]}"
    fi
done
```

## 3. Array Patterns: Stack, Queue, Set

### Stack (LIFO - Last In, First Out)

```bash
#!/usr/bin/env bash

# Stack implementation using array
declare -a stack

# Push
push() {
    stack+=("$1")
}

# Pop
pop() {
    if [ ${#stack[@]} -eq 0 ]; then
        return 1
    fi

    local value="${stack[-1]}"
    unset 'stack[-1]'
    echo "$value"
}

# Peek (view top without removing)
peek() {
    if [ ${#stack[@]} -eq 0 ]; then
        return 1
    fi
    echo "${stack[-1]}"
}

# Is empty
is_empty() {
    [ ${#stack[@]} -eq 0 ]
}

# Size
size() {
    echo "${#stack[@]}"
}

# Usage
push "first"
push "second"
push "third"

echo "Top: $(peek)"           # third
echo "Size: $(size)"          # 3
echo "Pop: $(pop)"            # third
echo "Pop: $(pop)"            # second
echo "Size: $(size)"          # 1
```

### Queue (FIFO - First In, First Out)

```bash
#!/usr/bin/env bash

# Queue implementation using array
declare -a queue

# Enqueue (add to end)
enqueue() {
    queue+=("$1")
}

# Dequeue (remove from front)
dequeue() {
    if [ ${#queue[@]} -eq 0 ]; then
        return 1
    fi

    local value="${queue[0]}"
    queue=("${queue[@]:1}")  # Remove first element
    echo "$value"
}

# Front (peek at first element)
front() {
    if [ ${#queue[@]} -eq 0 ]; then
        return 1
    fi
    echo "${queue[0]}"
}

# Usage
enqueue "job1"
enqueue "job2"
enqueue "job3"

echo "Front: $(front)"         # job1
echo "Dequeue: $(dequeue)"     # job1
echo "Dequeue: $(dequeue)"     # job2
echo "Front: $(front)"         # job3
```

### Set (Unique Values)

```bash
#!/usr/bin/env bash

# Set implementation using associative array
declare -A set

# Add element
set_add() {
    local value="$1"
    set[$value]=1
}

# Remove element
set_remove() {
    local value="$1"
    unset "set[$value]"
}

# Contains
set_contains() {
    local value="$1"
    [[ -v set[$value] ]]
}

# Size
set_size() {
    echo "${#set[@]}"
}

# Get all elements
set_elements() {
    echo "${!set[@]}"
}

# Union
set_union() {
    local -n set1=$1
    local -n set2=$2
    local -A result

    for key in "${!set1[@]}"; do
        result[$key]=1
    done

    for key in "${!set2[@]}"; do
        result[$key]=1
    done

    echo "${!result[@]}"
}

# Intersection
set_intersection() {
    local -n set1=$1
    local -n set2=$2
    local -A result

    for key in "${!set1[@]}"; do
        if [[ -v set2[$key] ]]; then
            result[$key]=1
        fi
    done

    echo "${!result[@]}"
}

# Usage
set_add "apple"
set_add "banana"
set_add "cherry"
set_add "apple"  # Duplicate, ignored

echo "Size: $(set_size)"      # 3
set_contains "banana" && echo "Contains banana"
echo "Elements: $(set_elements)"

# Set operations
declare -A set_a=(["a"]=1 ["b"]=1 ["c"]=1)
declare -A set_b=(["b"]=1 ["c"]=1 ["d"]=1)

echo "Union: $(set_union set_a set_b)"           # a b c d
echo "Intersection: $(set_intersection set_a set_b)"  # b c
```

## 4. CSV Parsing

Parse CSV (Comma-Separated Values) files into arrays.

### Basic CSV Reader

```bash
#!/usr/bin/env bash

# Read CSV into array
read_csv() {
    local file="$1"
    local -n rows_ref=$2

    [ -f "$file" ] || return 1

    local line
    while IFS= read -r line; do
        rows_ref+=("$line")
    done < "$file"
}

# Parse CSV line into array
parse_csv_line() {
    local line="$1"
    local -n fields_ref=$2

    local IFS=','
    read -ra fields_ref <<< "$line"
}

# Example CSV file
cat > data.csv <<'EOF'
name,age,city
John,30,New York
Jane,25,Los Angeles
Bob,35,Chicago
EOF

# Read and parse
declare -a rows
read_csv "data.csv" rows

echo "Total rows: ${#rows[@]}"

for row in "${rows[@]}"; do
    declare -a fields
    parse_csv_line "$row" fields
    echo "Fields: ${fields[@]}"
done

rm data.csv
```

### Advanced CSV Parser (Handles Quotes)

```bash
#!/usr/bin/env bash

# Parse CSV with quoted fields
parse_csv_advanced() {
    local line="$1"
    local -n result=$2

    result=()
    local field=""
    local in_quotes=false
    local i char

    for ((i=0; i<${#line}; i++)); do
        char="${line:i:1}"

        case "$char" in
            '"')
                if $in_quotes; then
                    # Check for escaped quote ""
                    if [[ "${line:i+1:1}" == '"' ]]; then
                        field+="$char"
                        ((i++))
                    else
                        in_quotes=false
                    fi
                else
                    in_quotes=true
                fi
                ;;
            ',')
                if $in_quotes; then
                    field+="$char"
                else
                    result+=("$field")
                    field=""
                fi
                ;;
            *)
                field+="$char"
                ;;
        esac
    done

    # Add last field
    result+=("$field")
}

# Test CSV with quotes
csv_line='John,"New York, NY","He said ""Hello"""'

declare -a fields
parse_csv_advanced "$csv_line" fields

echo "Number of fields: ${#fields[@]}"
for i in "${!fields[@]}"; do
    echo "Field $i: ${fields[$i]}"
done
# Field 0: John
# Field 1: New York, NY
# Field 2: He said "Hello"
```

### CSV Column Extraction

```bash
#!/usr/bin/env bash

# Extract specific column from CSV
extract_column() {
    local file="$1"
    local column_index="$2"
    local -n result=$3

    [ -f "$file" ] || return 1

    local line
    while IFS=, read -ra fields; do
        if [ ${#fields[@]} -gt $column_index ]; then
            result+=("${fields[$column_index]}")
        fi
    done < "$file"
}

# Example
cat > employees.csv <<'EOF'
name,department,salary
John,Engineering,80000
Jane,Marketing,75000
Bob,Engineering,85000
Alice,Sales,70000
EOF

# Extract departments (column 1)
declare -a departments
extract_column "employees.csv" 1 departments

echo "Departments:"
for dept in "${departments[@]}"; do
    echo "  $dept"
done

rm employees.csv
```

### CSV to Associative Array

```bash
#!/usr/bin/env bash

# Load CSV into array of associative arrays
load_csv_records() {
    local file="$1"
    local -n records_ref=$2

    [ -f "$file" ] || return 1

    # Read header
    local header
    IFS= read -r header < "$file"

    # Parse header
    local IFS=','
    read -ra headers <<< "$header"

    # Read data rows
    local line row_num=0
    while IFS= read -r line; do
        # Skip header
        [ $row_num -eq 0 ] && { ((row_num++)); continue; }

        # Parse row
        local IFS=','
        read -ra values <<< "$line"

        # Create record name
        local record_prefix="record_${row_num}"

        # Store each field
        for i in "${!headers[@]}"; do
            local key="${record_prefix}_${headers[$i]}"
            records_ref[$key]="${values[$i]}"
        done

        ((row_num++))
    done < "$file"

    # Return number of records
    echo $((row_num - 1))
}

# Example
cat > users.csv <<'EOF'
name,email,role
John,john@example.com,admin
Jane,jane@example.com,user
Bob,bob@example.com,user
EOF

declare -A records
num_records=$(load_csv_records "users.csv" records)

echo "Loaded $num_records records"

# Access records
for i in $(seq 1 $num_records); do
    echo "Record $i:"
    echo "  Name: ${records[record_${i}_name]}"
    echo "  Email: ${records[record_${i}_email]}"
    echo "  Role: ${records[record_${i}_role]}"
done

rm users.csv
```

## 5. Config File Loading

### Simple Key-Value Config

```bash
#!/usr/bin/env bash

# Load key=value config file
load_config() {
    local config_file="$1"
    local -n config_ref=$2

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

        # Must contain =
        [[ "$line" != *=* ]] && continue

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

        config_ref["$key"]="$value"
    done < "$config_file"
}

# Example config file
cat > app.conf <<'EOF'
# Application configuration
host = localhost
port = 8080
database = "myapp"

# Feature flags
debug = true
verbose = false  # Verbose logging
EOF

declare -A config
load_config "app.conf" config

echo "Configuration:"
for key in "${!config[@]}"; do
    echo "  $key = ${config[$key]}"
done

rm app.conf
```

### INI-Style Config with Sections

```bash
#!/usr/bin/env bash

# Load INI-style config with sections
load_ini_config() {
    local config_file="$1"
    local -n config_ref=$2

    [ -f "$config_file" ] || return 1

    local section=""
    local line key value

    while IFS= read -r line; do
        # Skip empty and comments
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        [[ "$line" =~ ^[[:space:]]*[#\;] ]] && continue

        # Trim whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"

        # Section header
        if [[ "$line" =~ ^\[.*\]$ ]]; then
            section="${line:1:-1}"
            continue
        fi

        # Key=value
        if [[ "$line" == *=* ]]; then
            key="${line%%=*}"
            value="${line#*=}"

            # Trim
            key="${key#"${key%%[![:space:]]*}"}"
            key="${key%"${key##*[![:space:]]}"}"
            value="${value#"${value%%[![:space:]]*}"}"
            value="${value%"${value##*[![:space:]]}"}"

            # Remove quotes
            if [[ "$value" =~ ^\".*\"$ ]] || [[ "$value" =~ ^\'.*\'$ ]]; then
                value="${value:1:-1}"
            fi

            # Store with section prefix
            if [ -n "$section" ]; then
                config_ref["${section}.${key}"]="$value"
            else
                config_ref["$key"]="$value"
            fi
        fi
    done < "$config_file"
}

# Example INI file
cat > config.ini <<'EOF'
; Global settings
timeout = 30

[database]
host = localhost
port = 5432
name = myapp

[cache]
host = localhost
port = 6379
ttl = 3600
EOF

declare -A config
load_ini_config "config.ini" config

echo "Configuration:"
for key in $(echo "${!config[@]}" | tr ' ' '\n' | sort); do
    echo "  $key = ${config[$key]}"
done

# Access specific sections
echo -e "\nDatabase config:"
for key in "${!config[@]}"; do
    if [[ "$key" == database.* ]]; then
        echo "  ${key#database.} = ${config[$key]}"
    fi
done

rm config.ini
```

### Config with Defaults

```bash
#!/usr/bin/env bash

# Config loader with default values
declare -A default_config=(
    [host]="localhost"
    [port]="8080"
    [debug]="false"
    [max_connections]="100"
    [timeout]="30"
)

load_config_with_defaults() {
    local config_file="$1"
    local -n config_ref=$2
    local -n defaults_ref=$3

    # Start with defaults
    for key in "${!defaults_ref[@]}"; do
        config_ref["$key"]="${defaults_ref[$key]}"
    done

    # Load config file if exists
    if [ -f "$config_file" ]; then
        local line key value
        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*$ ]] && continue
            [[ "$line" =~ ^[[:space:]]*# ]] && continue

            line="${line%%#*}"
            [[ "$line" != *=* ]] && continue

            key="${line%%=*}"
            value="${line#*=}"

            key="${key#"${key%%[![:space:]]*}"}"
            key="${key%"${key##*[![:space:]]}"}"
            value="${value#"${value%%[![:space:]]*}"}"
            value="${value%"${value##*[![:space:]]}"}"

            [[ "$value" =~ ^[\"\'].*[\"\']$ ]] && value="${value:1:-1}"

            config_ref["$key"]="$value"
        done < "$config_file"
    fi
}

# Test with partial config
cat > partial.conf <<'EOF'
host = production.example.com
debug = true
EOF

declare -A config
load_config_with_defaults "partial.conf" config default_config

echo "Final configuration:"
for key in "${!config[@]}"; do
    echo "  $key = ${config[$key]}"
done

rm partial.conf
```

## 6. Multi-Dimensional Data

Bash doesn't natively support multi-dimensional arrays, but we can simulate them.

### 2D Array Simulation

```bash
#!/usr/bin/env bash

# Simulate 2D array using naming convention: arr_row_col
declare -A matrix

# Set value at [row][col]
matrix_set() {
    local row=$1
    local col=$2
    local value=$3
    matrix["${row}_${col}"]="$value"
}

# Get value at [row][col]
matrix_get() {
    local row=$1
    local col=$2
    echo "${matrix["${row}_${col}"]}"
}

# Create 3x3 matrix
matrix_set 0 0 1
matrix_set 0 1 2
matrix_set 0 2 3
matrix_set 1 0 4
matrix_set 1 1 5
matrix_set 1 2 6
matrix_set 2 0 7
matrix_set 2 1 8
matrix_set 2 2 9

# Print matrix
echo "Matrix:"
for row in 0 1 2; do
    for col in 0 1 2; do
        printf "%3s" "$(matrix_get $row $col)"
    done
    echo
done
```

### Table Data Structure

```bash
#!/usr/bin/env bash

# Table with named columns
declare -A table
declare -a columns=("name" "age" "city")
declare -i row_count=0

# Add row
table_add_row() {
    local name="$1"
    local age="$2"
    local city="$3"

    table["${row_count}_name"]="$name"
    table["${row_count}_age"]="$age"
    table["${row_count}_city"]="$city"

    ((row_count++))
}

# Get cell value
table_get() {
    local row=$1
    local col=$2
    echo "${table["${row}_${col}"]}"
}

# Print table
table_print() {
    # Header
    printf "%-15s %-5s %-20s\n" "${columns[@]}"
    printf "%-15s %-5s %-20s\n" "---------------" "-----" "--------------------"

    # Rows
    for ((i=0; i<row_count; i++)); do
        printf "%-15s %-5s %-20s\n" \
            "$(table_get $i name)" \
            "$(table_get $i age)" \
            "$(table_get $i city)"
    done
}

# Usage
table_add_row "John Doe" 30 "New York"
table_add_row "Jane Smith" 25 "Los Angeles"
table_add_row "Bob Johnson" 35 "Chicago"

table_print
```

### Serialization with declare -p

```bash
#!/usr/bin/env bash

# Save array to file
save_array() {
    local array_name="$1"
    local file="$2"

    declare -p "$array_name" > "$file"
}

# Load array from file
load_array() {
    local file="$1"

    [ -f "$file" ] || return 1

    source "$file"
}

# Example with indexed array
my_array=(apple banana cherry)
save_array "my_array" "array.dat"

# Clear array
unset my_array

# Reload
load_array "array.dat"
echo "Restored: ${my_array[@]}"

# Example with associative array
declare -A my_map=([host]=localhost [port]=8080)
save_array "my_map" "map.dat"

# Clear and reload
unset my_map
load_array "map.dat"
echo "Restored: ${!my_map[@]}"

rm array.dat map.dat
```

## 7. Practical Patterns

### Argument Collection

```bash
#!/usr/bin/env bash

# Collect arguments into arrays
declare -a files
declare -a options
declare -A flags

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)
            flags[verbose]=1
            shift
            ;;
        -o|--output)
            flags[output]="$2"
            shift 2
            ;;
        --option=*)
            option="${1#*=}"
            options+=("$option")
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            files+=("$1")
            shift
            ;;
    esac
done

echo "Files: ${files[@]}"
echo "Options: ${options[@]}"
echo "Verbose: ${flags[verbose]:-0}"
echo "Output: ${flags[output]:-stdout}"
```

### Building Commands Dynamically

```bash
#!/usr/bin/env bash

# Build command with options
build_command() {
    local -a cmd=(rsync -av)

    # Add optional flags
    [ -n "$DRY_RUN" ] && cmd+=(--dry-run)
    [ -n "$DELETE" ] && cmd+=(--delete)
    [ -n "$COMPRESS" ] && cmd+=(--compress)

    # Add exclude patterns
    local -a excludes=(
        "*.tmp"
        "*.log"
        ".git"
    )

    for pattern in "${excludes[@]}"; do
        cmd+=(--exclude="$pattern")
    done

    # Add source and destination
    cmd+=("$SOURCE" "$DEST")

    # Execute
    echo "Running: ${cmd[@]}"
    "${cmd[@]}"
}

# Usage
SOURCE="/data"
DEST="/backup"
DRY_RUN=1
DELETE=1

build_command
```

### Safe Word Splitting

```bash
#!/usr/bin/env bash

# UNSAFE: word splitting on spaces
unsafe_cmd="ls -la /tmp"
$unsafe_cmd  # Vulnerable to injection

# SAFE: use array
safe_cmd=(ls -la /tmp)
"${safe_cmd[@]}"

# Building complex commands
build_find_command() {
    local dir="$1"
    shift

    local -a cmd=(find "$dir")

    # Add search criteria
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --name)
                cmd+=(-name "$2")
                shift 2
                ;;
            --type)
                cmd+=(-type "$2")
                shift 2
                ;;
            --newer)
                cmd+=(-newer "$2")
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    # Execute safely
    "${cmd[@]}"
}

# Usage
build_find_command "/var/log" --type f --name "*.log"
```

## Practice Problems

### Problem 1: Advanced CSV Processor

Create a CSV processing tool that:
- Parses CSV with quoted fields and escaped quotes
- Filters rows based on column values (regex support)
- Sorts by specified columns (multiple columns, asc/desc)
- Calculates aggregate functions (sum, avg, min, max, count)
- Joins two CSV files on common column
- Exports results in CSV or JSON format

**Example**:
```bash
csv_tool data.csv \
    --filter "age > 25" \
    --filter "city =~ ^New" \
    --sort "age:desc,name:asc" \
    --aggregate "avg(salary)" \
    --output result.csv
```

### Problem 2: Configuration Manager

Build a configuration management system that:
- Loads configs from multiple sources (files, environment, CLI args)
- Supports layered configs (defaults < system < user < environment)
- Validates config against schema (required fields, types, ranges)
- Provides get/set/list operations
- Exports configs in different formats (INI, JSON, YAML-like)
- Watches config files for changes

**Example**:
```bash
config load defaults.conf system.conf user.conf
config set database.host localhost
config get database.port  # Returns with fallback to defaults
config validate           # Check all required fields
config export --format json > config.json
```

### Problem 3: Data Structure Library

Implement a library of data structures:
- **Stack**: push, pop, peek, size, clear
- **Queue**: enqueue, dequeue, front, back, size
- **Priority Queue**: insert with priority, extract_max
- **Set**: add, remove, contains, union, intersection, difference
- **Map**: put, get, remove, keys, values, entries
- **List**: append, prepend, insert, remove, get, size

Include persistence (save/load to files) for each structure.

### Problem 4: Table Data Processor

Create a table manipulation tool with:
- Load data from CSV/TSV/fixed-width files
- Add/remove/rename columns
- Filter rows with complex conditions
- Transform values (map function over columns)
- Group by columns with aggregation
- Pivot/unpivot operations
- Export to various formats

**Example**:
```bash
table load employees.csv
table filter 'salary > 50000'
table group_by department --aggregate 'avg(salary),count(*)'
table sort salary:desc
table select name,department,salary
table export --format markdown
```

### Problem 5: Log Aggregator

Write a log aggregation tool that:
- Parses logs from multiple files (different formats)
- Extracts structured data (timestamp, level, message, metadata)
- Stores in searchable data structure
- Filters by time range, level, pattern
- Groups and counts by various fields
- Generates statistics (error rate, patterns, top errors)
- Exports reports in different formats

**Example**:
```bash
log_agg --input "*.log" \
    --format apache \
    --time-range "2024-02-13 00:00 to 2024-02-13 23:59" \
    --filter "level=ERROR" \
    --group-by hour \
    --top errors 10 \
    --output report.html
```

---

**Previous**: [02_Parameter_Expansion.md](./02_Parameter_Expansion.md) | **Next**: [04_Advanced_Control_Flow.md](./04_Advanced_Control_Flow.md)
