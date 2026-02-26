# 레슨 05: 함수와 라이브러리(Functions and Libraries)

**난이도**: ⭐⭐

**이전**: [고급 제어 흐름](./04_Advanced_Control_Flow.md) | **다음**: [I/O와 리다이렉션](./06_IO_and_Redirection.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 반환 값 패턴(echo 캡처, 전역 변수, nameref, 상태 코드)을 비교하고 각 상황에 맞는 패턴을 선택할 수 있습니다
2. 트리 순회와 메모이제이션(memoized) 계산을 위한 재귀 함수를 bash로 작성할 수 있습니다
3. 올바른 소싱(sourcing)과 경로 해석을 갖춘 재사용 가능한 함수 라이브러리를 만들 수 있습니다
4. 라이브러리 간 함수 및 변수 충돌을 방지하기 위한 네임스페이싱(namespacing) 규칙을 적용할 수 있습니다
5. 확장 가능한 스크립트 설계를 위해 콜백(callback)과 이벤트 핸들러(event-handler) 패턴을 구현할 수 있습니다
6. bash의 동적 스코핑(dynamic scoping) 규칙을 설명하고 흔한 변수 스코프 함정을 피할 수 있습니다
7. 모범 사례 템플릿에 따라 문서화와 입력 유효성 검사를 갖춘 함수를 작성할 수 있습니다

---

스크립트가 수백 줄을 넘어서면, 어떤 소프트웨어 프로젝트에서도 요구하는 것과 동일한 모듈식 구조가 필요해집니다: 재사용 가능한 함수, 공유 라이브러리, 명확한 인터페이스, 예측 가능한 스코핑(scoping). Bash 함수는 대부분의 프로그래밍 언어의 함수와 다르게 동작합니다 -- 진정한 반환 값이 없고, 동적 스코핑을 사용하며, 기본적으로 전역 네임스페이스를 공유합니다. 이러한 메커니즘을 이해하는 것은 유지 관리 가능한 자동화 툴킷을 구축하는 데 필수적입니다.

## 1. 반환 값 패턴(Return Value Patterns)

Bash 함수는 전통적인 프로그래밍 언어처럼 값을 반환하지 않습니다. 대신 결과를 호출자에게 전달하기 위해 여러 패턴을 사용합니다.

### 1.1 Echo 캡처 패턴(Echo Capture Pattern)

가장 일반적인 패턴은 결과를 `echo`하고 명령 치환으로 캡처하는 것입니다:

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

**장점**: 깔끔하고 함수형 스타일; 여러 echo 문을 통해 여러 반환 값 지원.

**단점**: 서브셸 생성으로 인해 느림; stdout과 반환 값을 구분할 수 없음.

### 1.2 전역 변수 패턴(Global Variable Pattern)

함수는 전역 변수를 직접 수정할 수 있습니다:

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

**장점**: 빠름; 여러 값을 쉽게 반환; 서브셸 오버헤드 없음.

**단점**: 전역 네임스페이스 오염; 추론하기 어려움; 스레드 안전하지 않음.

### 1.3 Nameref 패턴 (Bash 4.3+)

`declare -n`을 사용하여 변수에 대한 참조를 생성합니다:

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

**장점**: 깨끗한 분리; 전역 오염 없음; 호출자의 변수를 직접 수정 가능.

**단점**: Bash 4.3+ 필요; 약간 복잡한 구문.

### 1.4 반환 상태 코드 패턴(Return Status Code Pattern)

`return`을 사용하여 종료 상태(0-255)를 설정합니다:

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

**장점**: 표준 Unix 관례; 성공/실패 확인에 적합.

**단점**: 정수 0-255로 제한됨; 종종 다른 패턴과 결합됨.

### 1.5 비교 표(Comparison Table)

| 패턴 | 속도 | 다중 반환 | 복잡도 | 최적 사용 사례 |
|---------|-------|------------------|------------|---------------|
| Echo Capture | Slow | Yes (multiple echoes) | Low | Simple value returns |
| Global Variable | Fast | Yes | Medium | Performance-critical code |
| Nameref | Fast | Yes | Medium | Clean API design |
| Return Status | Fast | No | Low | Success/failure checks |

## 2. 재귀 함수(Recursive Functions)

재귀 함수는 재귀적 구조를 가진 문제를 해결하기 위해 자신을 호출합니다.

### 2.1 팩토리얼(Factorial)

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

### 2.2 디렉토리 트리 탐색(Directory Tree Traversal)

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

### 2.3 메모이제이션을 활용한 피보나치(Fibonacci with Memoization)

메모이제이션 없이는 재귀 피보나치가 매우 느립니다. 다음은 최적화된 버전입니다:

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

**출력**:
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

## 3. 함수 라이브러리(Function Libraries)

재사용 가능한 함수를 별도의 라이브러리 파일로 구성하여 유지보수성과 재사용성을 높입니다.

### 3.1 라이브러리 파일 생성(Creating a Library File)

**파일: `/opt/mylibs/string_utils.sh`**

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

### 3.2 라이브러리 파일 소싱(Sourcing Library Files)

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

### 3.3 라이브러리 경로 관리(Library Path Management)

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

### 3.4 초기화가 있는 라이브러리(Library with Initialization)

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

## 4. 네임스페이싱(Namespacing)

Bash는 내장 네임스페이스가 없지만 명명 규칙으로 시뮬레이션할 수 있습니다.

### 4.1 접두사 함수 이름(Prefixed Function Names)

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

### 4.2 네임스페이스 헬퍼 함수(Namespace Helper Functions)

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

## 5. 콜백 패턴(Callback Patterns)

함수 이름을 인수로 전달하여 유연하고 이벤트 기반 코드를 만듭니다.

### 5.1 간단한 콜백(Simple Callback)

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

### 5.2 이벤트 핸들러 패턴(Event Handler Pattern)

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

### 5.3 필터와 맵 패턴(Filter and Map Pattern)

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

## 6. 변수 스코프(Variable Scope)

변수 스코프(Variable Scope)를 이해하는 것은 올바른 함수를 작성하는 데 중요합니다.

### 6.1 지역 변수 vs 전역 변수(Local vs Global Variables)

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

**출력**:
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

### 6.2 Bash의 동적 스코핑(Dynamic Scoping in Bash)

Bash는 동적 스코핑(Dynamic Scoping)을 사용하며 렉시컬 스코핑(Lexical Scoping)이 아닙니다:

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

### 6.3 지역 Nameref(Local Nameref)

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

### 6.4 일반적인 함정 피하기(Avoiding Common Pitfalls)

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

## 7. 함수 모범 사례(Function Best Practices)

### 7.1 문서화 주석(Documentation Comments)

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

### 7.2 입력 검증(Input Validation)

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

### 7.3 함수에서의 에러 처리(Error Handling in Functions)

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

### 7.4 함수 템플릿(Function Template)

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

## 연습 문제(Practice Problems)

### 문제 1: 문자열 유틸리티 라이브러리
다음 함수들을 갖춘 문자열 유틸리티 라이브러리(`string_lib.sh`)를 만드세요:
- `str_reverse()` - 문자열 뒤집기
- `str_is_palindrome()` - 문자열이 회문인지 확인 (true일 경우 0 반환)
- `str_count_words()` - 문자열의 단어 개수 세기
- `str_capitalize()` - 각 단어의 첫 글자를 대문자로
- `str_remove_duplicates()` - 연속된 중복 문자 제거

별도의 스크립트로 라이브러리를 테스트하세요.

### 문제 2: 재귀적 파일 검색
다음 기능을 갖춘 재귀 함수 `find_files()`를 작성하세요:
- 디렉토리 경로와 파일 패턴(예: "*.txt")을 받음
- 모든 하위 디렉토리를 재귀적으로 검색
- nameref를 통해 일치하는 파일 경로 배열 반환
- 심볼릭 링크를 안전하게 처리 (무한 루프 방지)
- 검색된 총 파일 수와 발견된 일치 항목 수 계산

### 문제 3: 콜백을 사용한 계산기
다음 기능을 갖춘 계산기를 만드세요:
- 각 연산(+, -, *, /)에 대한 콜백 함수 허용
- `register_op()`를 통한 사용자 정의 연산 등록 지원
- 입력 검증 및 오류 처리
- 연산 히스토리 유지
- 예제: `calc 10 "+" 5` → 등록된 덧셈 콜백 사용

### 문제 4: 설정 관리자
다음 기능을 갖춘 설정 관리 시스템을 만드세요:
- `config_load()` - 파일에서 연관 배열(Associative Array)로 설정 로드
- `config_get()` - 기본값 폴백과 함께 값 가져오기
- `config_set()` - 검증 콜백과 함께 값 설정
- `config_save()` - 설정을 파일로 다시 저장
- `config_watch()` - 파일 변경 시 설정 다시 로드 (변경 시 콜백)

네임스페이싱(config::*)과 적절한 오류 처리를 사용하세요.

### 문제 5: 함수 성능 프로파일러
다음 기능을 갖춘 프로파일러를 작성하세요:
- 실행 시간을 측정하기 위해 모든 함수를 래핑
- 각 함수의 호출 횟수 추적
- 최소/최대/평균 실행 시간 기록
- 성능 보고서 생성
- 예제: `profile my_function arg1 arg2`

**이전**: [04_Advanced_Control_Flow.md](./04_Advanced_Control_Flow.md) | **다음**: [06_IO_and_Redirection.md](./06_IO_and_Redirection.md)
