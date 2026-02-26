# 배열과 데이터 구조 ⭐⭐

**이전**: [매개변수 확장과 변수 속성](./02_Parameter_Expansion.md) | **다음**: [고급 제어 흐름](./04_Advanced_Control_Flow.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 공백을 안전하게 처리하기 위한 올바른 인용(quoting)을 사용하여 인덱스 배열을 선언, 채우기, 순회할 수 있습니다
2. 키-값 데이터 저장을 위한 연관 배열(associative array)을 생성하고 조작할 수 있습니다
3. bash 배열을 사용해 스택(stack), 큐(queue), 집합(set) 패턴을 구현할 수 있습니다
4. 올바른 필드 분리와 인용을 적용해 CSV 데이터를 줄 단위로 파싱할 수 있습니다
5. key=value 파일에서 연관 배열을 채우는 설정 파일 로더를 작성할 수 있습니다
6. 복합 키(compound key) 또는 병렬 배열을 사용해 다차원 데이터 구조를 시뮬레이션할 수 있습니다
7. 배치 파일 이름 변경, 옵션 파싱, 데이터 집계에 배열 기반 패턴을 적용할 수 있습니다

---

실제 스크립트는 단일 스칼라 값만 처리하는 경우가 거의 없습니다. 서버 목록을 처리하고, 설정 파일을 파싱하고, 명령 출력을 수집하고, 구조화된 레코드를 관리해야 합니다. Bash 배열(array)을 사용하면 임시 파일을 작성하거나 외부 도구를 호출하지 않고도 이러한 작업을 수행할 수 있습니다. 배열을 숙달하는 것이 취약한 일회용 스크립트와 유지 관리 가능한 자동화의 차이를 만듭니다.

## 1. 인덱스 배열 연산

인덱스 배열(Indexed Arrays)은 0부터 시작하는 숫자 인덱스로 요소를 저장합니다.

### 배열 선언 및 초기화

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

### 배열 요소 접근

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

### 배열 수정

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

### 배열 슬라이싱

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

### 반복

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

### 배열 복사 및 병합

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

## 2. 연관 배열

연관 배열(Associative Arrays, 해시맵, 딕셔너리)은 숫자 인덱스 대신 문자열을 키로 사용합니다. Bash 4.0+ 이상에서 사용 가능합니다.

### 선언 및 기본 연산

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

### 반복

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

### 비교: 인덱스 배열 vs 연관 배열

| 기능 | 인덱스 배열 | 연관 배열 |
|---------|---------------|-------------------|
| 선언 | `arr=()` 또는 `declare -a arr` | `declare -A arr` |
| 키 | 정수 (0, 1, 2, ...) | 문자열 |
| 순서 | 보존됨 | 정렬되지 않음 |
| 사용 가능 버전 | 모든 bash 버전 | Bash 4.0+ |
| 사용 사례 | 순차 데이터 | 키-값 쌍 |
| 반복 | 순서 보장됨 | 순서 정의되지 않음 |
| 접근 | `${arr[0]}` | `${arr[key]}` |
| 모든 키 | `${!arr[@]}` | `${!arr[@]}` |

### 실용 예제

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

### 중첩 데이터 시뮬레이션

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

## 3. 배열 패턴: 스택, 큐, 집합

### 스택 (LIFO - Last In, First Out)

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

### 큐 (FIFO - First In, First Out)

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

### 집합 (고유 값)

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

## 4. CSV 파싱

CSV(Comma-Separated Values) 파일을 배열로 파싱합니다.

### 기본 CSV 리더

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

### 고급 CSV 파서 (따옴표 처리)

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

### CSV 열 추출

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

### CSV를 연관 배열로 변환

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

## 5. 설정 파일 로딩

### 간단한 키-값 설정

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

### 섹션이 있는 INI 스타일 설정

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

### 기본값이 있는 설정

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

## 6. 다차원 데이터

Bash는 기본적으로 다차원 배열을 지원하지 않지만 시뮬레이션할 수 있습니다.

### 2차원 배열 시뮬레이션

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

### 테이블 데이터 구조

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

### declare -p를 사용한 직렬화

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

## 7. 실용 패턴

### 인수 수집

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

### 동적으로 명령 구성

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

### 안전한 단어 분할

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

## 연습 문제

### 문제 1: 고급 CSV 처리기

다음 기능을 가진 CSV 처리 도구를 만드세요:
- 따옴표가 있는 필드와 이스케이프된 따옴표가 있는 CSV 파싱
- 열 값에 기반한 행 필터링 (정규식 지원)
- 지정된 열로 정렬 (여러 열, 오름차순/내림차순)
- 집계 함수 계산 (sum, avg, min, max, count)
- 공통 열을 기준으로 두 CSV 파일 조인
- CSV 또는 JSON 형식으로 결과 내보내기

**예제**:
```bash
csv_tool data.csv \
    --filter "age > 25" \
    --filter "city =~ ^New" \
    --sort "age:desc,name:asc" \
    --aggregate "avg(salary)" \
    --output result.csv
```

### 문제 2: 설정 관리자

다음 기능을 가진 설정 관리 시스템을 구축하세요:
- 여러 소스에서 설정 로드 (파일, 환경, CLI 인수)
- 계층화된 설정 지원 (기본값 < 시스템 < 사용자 < 환경)
- 스키마에 대한 설정 검증 (필수 필드, 타입, 범위)
- get/set/list 연산 제공
- 다양한 형식으로 설정 내보내기 (INI, JSON, YAML-like)
- 설정 파일 변경 감시

**예제**:
```bash
config load defaults.conf system.conf user.conf
config set database.host localhost
config get database.port  # Returns with fallback to defaults
config validate           # Check all required fields
config export --format json > config.json
```

### 문제 3: 데이터 구조 라이브러리

다음 데이터 구조들의 라이브러리를 구현하세요:
- **스택(Stack)**: push, pop, peek, size, clear
- **큐(Queue)**: enqueue, dequeue, front, back, size
- **우선순위 큐(Priority Queue)**: insert with priority, extract_max
- **집합(Set)**: add, remove, contains, union, intersection, difference
- **맵(Map)**: put, get, remove, keys, values, entries
- **리스트(List)**: append, prepend, insert, remove, get, size

각 구조에 대한 영속성(파일로 저장/로드) 포함하세요.

### 문제 4: 테이블 데이터 처리기

다음 기능을 가진 테이블 조작 도구를 만드세요:
- CSV/TSV/고정 너비 파일에서 데이터 로드
- 열 추가/제거/이름 변경
- 복잡한 조건으로 행 필터링
- 값 변환 (열에 대한 맵 함수)
- 집계와 함께 열별로 그룹화
- 피벗/언피벗 연산
- 다양한 형식으로 내보내기

**예제**:
```bash
table load employees.csv
table filter 'salary > 50000'
table group_by department --aggregate 'avg(salary),count(*)'
table sort salary:desc
table select name,department,salary
table export --format markdown
```

### 문제 5: 로그 집계기

다음 기능을 가진 로그 집계 도구를 작성하세요:
- 여러 파일에서 로그 파싱 (다양한 형식)
- 구조화된 데이터 추출 (타임스탬프, 레벨, 메시지, 메타데이터)
- 검색 가능한 데이터 구조에 저장
- 시간 범위, 레벨, 패턴으로 필터링
- 다양한 필드별로 그룹화 및 카운트
- 통계 생성 (오류율, 패턴, 상위 오류)
- 다양한 형식으로 보고서 내보내기

**예제**:
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

**이전**: [02_Parameter_Expansion.md](./02_Parameter_Expansion.md) | **다음**: [04_Advanced_Control_Flow.md](./04_Advanced_Control_Flow.md)
