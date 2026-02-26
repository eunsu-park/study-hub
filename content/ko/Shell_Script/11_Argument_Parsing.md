# Lesson 11: 인수 파싱 및 CLI 인터페이스

**난이도**: ⭐⭐⭐

**이전**: [에러 처리 및 디버깅](./10_Error_Handling.md) | **다음**: [이식성과 모범 사례](./12_Portability_and_Best_Practices.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `while`/`case` 루프를 사용해 짧은 옵션, 긴 옵션, 결합 옵션을 지원하는 수동 인수 파싱(manual argument parsing)을 구현할 수 있다
2. POSIX `getopts`를 적용해 조용한 모드(silent mode)에서 적절한 에러 처리와 함께 짧은 옵션을 파싱할 수 있다
3. `getopts`(POSIX)와 GNU `getopt`를 비교하고, 주어진 이식성 요건에 맞는 도구를 선택할 수 있다
4. NAME/SYNOPSIS/DESCRIPTION/OPTIONS 관례를 따르는 자기 문서화(self-documenting) 도움말 메시지를 작성할 수 있다
5. 터미널 기능 자동 감지 및 `NO_COLOR` 지원과 함께 색상 출력(color output)을 구현할 수 있다
6. 스피너(spinner), 진행 막대(progress bar), 다중 작업 대시보드를 포함한 진행 표시기(progress indicator)를 제작할 수 있다
7. 유효성 검사, 메뉴, 비밀번호 마스킹이 포함된 대화형 입력 프롬프트(interactive input prompt)를 설계할 수 있다
8. 인수 파싱, 색상 출력, 진행 표시를 결합해 전문적인 CLI 도구를 만들 수 있다

---

사용자 입력을 받는 모든 스크립트에는 커맨드라인 인수(command-line argument)를 파싱하는 방법이 필요합니다. 명확한 도움말 텍스트, 직관적인 플래그, 유익한 진행 피드백을 갖춘 잘 설계된 CLI 인터페이스는 스크립트가 세련된 도구처럼 느껴지는지 아니면 불안정한 즉흥 작업처럼 느껴지는지를 결정합니다. 팀원과 스크립트를 공유하거나 CI/CD 파이프라인에서 실행할 때, 적절한 인수 처리는 "내 컴퓨터에서는 작동했는데"와 신뢰할 수 있는 자기 문서화 자동화 사이의 차이를 만들어 냅니다.

## 1. 수동 인수 파싱

수동 파싱은 인수 처리에 대한 완전한 제어를 제공합니다.

### 기본 인수 루프

```bash
#!/bin/bash

# Parse arguments manually
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            # Positional argument
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Remaining arguments (after --)
REMAINING_ARGS=("$@")

echo "VERBOSE: ${VERBOSE:-0}"
echo "OUTPUT: ${OUTPUT:-none}"
echo "ARGS: ${ARGS[*]}"
echo "REMAINING: ${REMAINING_ARGS[*]}"
```

### 값을 가진 옵션 처리

```bash
#!/bin/bash

# Parse options that take values
parse_args() {
    local verbose=0
    local output=""
    local count=1
    local files=()

    while [ $# -gt 0 ]; do
        case "$1" in
            -v|--verbose)
                verbose=1
                shift
                ;;
            -o|--output)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "Error: --output requires a value" >&2
                    return 1
                fi
                output="$2"
                shift 2
                ;;
            --output=*)
                output="${1#*=}"
                shift
                ;;
            -n|--count)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "Error: --count requires a value" >&2
                    return 1
                fi
                count="$2"
                shift 2
                ;;
            --count=*)
                count="${1#*=}"
                shift
                ;;
            --)
                shift
                files=("$@")
                break
                ;;
            -*)
                echo "Unknown option: $1" >&2
                return 1
                ;;
            *)
                files+=("$1")
                shift
                ;;
        esac
    done

    # Export parsed values
    echo "verbose=$verbose"
    echo "output=$output"
    echo "count=$count"
    echo "files=(${files[*]})"
}

# Test
parse_args -v --output=result.txt --count 5 file1.txt file2.txt
echo "---"
parse_args --verbose -o result.txt -n 3 -- file1.txt file2.txt -special-file
```

### 고급 수동 파싱

```bash
#!/bin/bash

# Complete argument parser
declare -A OPTIONS
declare -a POSITIONAL

parse_arguments() {
    local expecting_value=""
    local option_name=""

    while [ $# -gt 0 ]; do
        # Handle value for previous option
        if [ -n "$expecting_value" ]; then
            OPTIONS["$option_name"]="$1"
            expecting_value=""
            option_name=""
            shift
            continue
        fi

        case "$1" in
            # Long option with value: --option=value
            --*=*)
                option_name="${1%%=*}"
                option_name="${option_name#--}"
                OPTIONS["$option_name"]="${1#*=}"
                shift
                ;;

            # Long option without value: --option
            --*)
                option_name="${1#--}"
                # Check if next arg is a value or another option
                if [ $# -gt 1 ] && [[ ! "$2" =~ ^- ]]; then
                    expecting_value=1
                else
                    OPTIONS["$option_name"]=1
                fi
                shift
                ;;

            # Short option: -o
            -[!-])
                option_name="${1#-}"
                # Check if next arg is a value
                if [ $# -gt 1 ] && [[ ! "$2" =~ ^- ]]; then
                    expecting_value=1
                else
                    OPTIONS["$option_name"]=1
                fi
                shift
                ;;

            # Combined short options: -abc
            -[!-]*)
                local opts="${1#-}"
                for (( i=0; i<${#opts}; i++ )); do
                    OPTIONS["${opts:$i:1}"]=1
                done
                shift
                ;;

            # End of options
            --)
                shift
                POSITIONAL+=("$@")
                break
                ;;

            # Positional argument
            *)
                POSITIONAL+=("$1")
                shift
                ;;
        esac
    done

    # Check if we're still expecting a value
    if [ -n "$expecting_value" ]; then
        echo "Error: Option --$option_name requires a value" >&2
        return 1
    fi
}

# Usage
parse_arguments -abc --verbose --output=file.txt --count 5 input1.txt input2.txt

# Display results
echo "Options:"
for key in "${!OPTIONS[@]}"; do
    echo "  $key = ${OPTIONS[$key]}"
done

echo "Positional arguments:"
for arg in "${POSITIONAL[@]}"; do
    echo "  $arg"
done
```

## 2. getopts (POSIX)

`getopts`는 옵션 파싱을 위한 POSIX 내장 명령어입니다.

### 기본 getopts 사용법

```bash
#!/bin/bash

# Parse options with getopts
usage() {
    echo "Usage: $0 [-v] [-o OUTPUT] [-n COUNT] FILE..."
    exit 1
}

verbose=0
output=""
count=1

while getopts "vo:n:h" opt; do
    case "$opt" in
        v)
            verbose=1
            ;;
        o)
            output="$OPTARG"
            ;;
        n)
            count="$OPTARG"
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument" >&2
            usage
            ;;
    esac
done

# Shift processed options
shift $((OPTIND - 1))

# Remaining arguments are positional
files=("$@")

echo "verbose=$verbose"
echo "output=$output"
echo "count=$count"
echo "files=(${files[*]})"
```

### getopts 에러 처리

```bash
#!/bin/bash

# Two error handling modes:
# 1. Default (verbose): getopts prints errors
# 2. Silent mode: prepend option string with ":"

# Silent mode (recommended)
while getopts ":vho:n:" opt; do
    case "$opt" in
        v)
            VERBOSE=1
            ;;
        o)
            OUTPUT="$OPTARG"
            ;;
        n)
            COUNT="$OPTARG"
            # Validate it's a number
            if ! [[ "$COUNT" =~ ^[0-9]+$ ]]; then
                echo "Error: -n requires a number" >&2
                exit 1
            fi
            ;;
        h)
            echo "Help message"
            exit 0
            ;;
        :)
            # Option requires argument but none provided
            echo "Error: -$OPTARG requires an argument" >&2
            exit 1
            ;;
        \?)
            # Invalid option
            echo "Error: Invalid option -$OPTARG" >&2
            exit 1
            ;;
    esac
done

shift $((OPTIND - 1))

echo "Parsed successfully"
echo "Remaining args: $*"
```

### 함수와 함께 getopts 사용

```bash
#!/bin/bash

# Parse options in a function
parse_options() {
    local OPTIND opt
    local verbose=0
    local output=""

    while getopts ":vo:" opt; do
        case "$opt" in
            v) verbose=1 ;;
            o) output="$OPTARG" ;;
            \?) echo "Invalid option: -$OPTARG" >&2; return 1 ;;
            :) echo "Option -$OPTARG requires an argument" >&2; return 1 ;;
        esac
    done

    shift $((OPTIND - 1))

    # Return parsed values (using global variables or output)
    PARSED_VERBOSE=$verbose
    PARSED_OUTPUT=$output
    PARSED_ARGS=("$@")
}

# Call parser
parse_options -v -o output.txt file1 file2

echo "verbose=$PARSED_VERBOSE"
echo "output=$PARSED_OUTPUT"
echo "args=${PARSED_ARGS[*]}"
```

### 완전한 getopts 예제

```bash
#!/bin/bash

set -euo pipefail

# Script configuration
SCRIPT_NAME=$(basename "$0")
VERSION="1.0.0"

# Default values
VERBOSE=0
DRY_RUN=0
OUTPUT_FILE=""
INPUT_FILES=()

# Usage message
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS] FILE...

Process files with various options.

OPTIONS:
    -v          Verbose mode
    -n          Dry run (don't make changes)
    -o FILE     Output file
    -h          Show this help message
    -V          Show version

EXAMPLES:
    $SCRIPT_NAME -v input.txt
    $SCRIPT_NAME -o output.txt -n input1.txt input2.txt
EOF
    exit 0
}

# Version message
version() {
    echo "$SCRIPT_NAME version $VERSION"
    exit 0
}

# Parse options
while getopts ":vno:hV" opt; do
    case "$opt" in
        v)
            VERBOSE=1
            ;;
        n)
            DRY_RUN=1
            ;;
        o)
            OUTPUT_FILE="$OPTARG"
            ;;
        h)
            usage
            ;;
        V)
            version
            ;;
        :)
            echo "Error: Option -$OPTARG requires an argument" >&2
            echo "Try '$SCRIPT_NAME -h' for more information." >&2
            exit 1
            ;;
        \?)
            echo "Error: Invalid option -$OPTARG" >&2
            echo "Try '$SCRIPT_NAME -h' for more information." >&2
            exit 1
            ;;
    esac
done

shift $((OPTIND - 1))

# Validate arguments
if [ $# -eq 0 ]; then
    echo "Error: No input files specified" >&2
    echo "Try '$SCRIPT_NAME -h' for more information." >&2
    exit 1
fi

INPUT_FILES=("$@")

# Process files
[ $VERBOSE -eq 1 ] && echo "Processing ${#INPUT_FILES[@]} files..."
[ $DRY_RUN -eq 1 ] && echo "DRY RUN MODE"

for file in "${INPUT_FILES[@]}"; do
    [ $VERBOSE -eq 1 ] && echo "Processing: $file"
    # Process file here
done

[ -n "$OUTPUT_FILE" ] && echo "Output: $OUTPUT_FILE"
```

## 3. getopt (GNU)

GNU `getopt`는 긴 옵션과 더 고급 파싱을 지원합니다.

### 기본 getopt 사용법

```bash
#!/bin/bash

# Note: This requires GNU getopt (not available on macOS by default)
# macOS users: brew install gnu-getopt

# Parse with getopt
OPTS=$(getopt -o "vo:n:" --long "verbose,output:,count:" -n "$0" -- "$@")

if [ $? -ne 0 ]; then
    echo "Failed to parse options" >&2
    exit 1
fi

# Reset positional parameters
eval set -- "$OPTS"

# Parse options
verbose=0
output=""
count=1

while true; do
    case "$1" in
        -v|--verbose)
            verbose=1
            shift
            ;;
        -o|--output)
            output="$2"
            shift 2
            ;;
        -n|--count)
            count="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!" >&2
            exit 1
            ;;
    esac
done

# Remaining arguments
files=("$@")

echo "verbose=$verbose"
echo "output=$output"
echo "count=$count"
echo "files=(${files[*]})"
```

### 긴 옵션만 사용하는 getopt

```bash
#!/bin/bash

# Long options only
OPTS=$(getopt --long "help,version,verbose,output:,dry-run" -n "$0" -- "$@")

if [ $? -ne 0 ]; then
    exit 1
fi

eval set -- "$OPTS"

while true; do
    case "$1" in
        --help)
            echo "Help message"
            exit 0
            ;;
        --version)
            echo "Version 1.0.0"
            exit 0
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!" >&2
            exit 1
            ;;
    esac
done

echo "Parsed options successfully"
```

### getopt vs getopts 비교

| 기능 | getopts (POSIX) | getopt (GNU) |
|---------|----------------|--------------|
| 이식성 | POSIX (모든 시스템) | GNU (Linux, 설치된 macOS) |
| 긴 옵션 | 아니오 | 예 |
| 옵션 묶기 | 제한적 | 완전 지원 |
| `--` 구분자 | 수동 처리 | 내장 |
| 에러 메시지 | 기본 | 상세 |
| 옵션 재정렬 | 아니오 | 예 |
| 복잡도 | 간단 | 더 복잡 |
| 사용 사례 | 간단한 스크립트 | 복잡한 CLI 도구 |

### 완전한 getopt 예제

```bash
#!/bin/bash

set -euo pipefail

SCRIPT_NAME=$(basename "$0")

# Check if GNU getopt is available
if ! getopt --test > /dev/null 2>&1; then
    if [ $? -ne 4 ]; then
        echo "Error: GNU getopt not available" >&2
        exit 1
    fi
fi

# Parse options
SHORT_OPTS="vno:h"
LONG_OPTS="verbose,dry-run,output:,help,version,config:"

OPTS=$(getopt -o "$SHORT_OPTS" --long "$LONG_OPTS" -n "$SCRIPT_NAME" -- "$@")

if [ $? -ne 0 ]; then
    echo "Run '$SCRIPT_NAME --help' for usage" >&2
    exit 1
fi

eval set -- "$OPTS"

# Default values
VERBOSE=0
DRY_RUN=0
OUTPUT=""
CONFIG=""

# Parse
while true; do
    case "$1" in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            if [ ! -f "$CONFIG" ]; then
                echo "Error: Config file not found: $CONFIG" >&2
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: $SCRIPT_NAME [OPTIONS] FILE...

OPTIONS:
    -v, --verbose       Verbose output
    -n, --dry-run       Dry run mode
    -o, --output FILE   Output file
    --config FILE       Configuration file
    -h, --help          Show this help
    --version           Show version
EOF
            exit 0
            ;;
        --version)
            echo "$SCRIPT_NAME 1.0.0"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!" >&2
            exit 1
            ;;
    esac
done

# Remaining arguments
FILES=("$@")

if [ ${#FILES[@]} -eq 0 ]; then
    echo "Error: No input files specified" >&2
    exit 1
fi

# Execute
[ $VERBOSE -eq 1 ] && echo "Processing ${#FILES[@]} files"
[ $DRY_RUN -eq 1 ] && echo "DRY RUN MODE"

for file in "${FILES[@]}"; do
    [ $VERBOSE -eq 1 ] && echo "Processing: $file"
done
```

## 4. 자체 문서화 도움말

좋은 도움말 메시지는 CLI 도구를 사용자 친화적으로 만듭니다.

### 도움말 메시지 템플릿

```bash
#!/bin/bash

show_help() {
    cat << EOF
NAME
    $(basename "$0") - Brief description of what the script does

SYNOPSIS
    $(basename "$0") [OPTIONS] COMMAND [ARGUMENTS]

DESCRIPTION
    Detailed description of what this script does.
    Can span multiple lines and include examples.

OPTIONS
    -v, --verbose
        Enable verbose output

    -o, --output FILE
        Specify output file (default: stdout)

    -n, --count NUMBER
        Number of iterations (default: 1)

    -h, --help
        Show this help message and exit

    -V, --version
        Show version information and exit

COMMANDS
    start       Start the service
    stop        Stop the service
    restart     Restart the service
    status      Show service status

EXAMPLES
    # Basic usage
    $(basename "$0") start

    # With options
    $(basename "$0") -v --output=log.txt start

    # Multiple operations
    $(basename "$0") -n 5 process file1.txt file2.txt

EXIT STATUS
    0   Success
    1   General error
    2   Invalid arguments
    66  Input file not found
    77  Permission denied

AUTHOR
    Written by Your Name

REPORTING BUGS
    Report bugs to: bugs@example.com

SEE ALSO
    Full documentation at: https://example.com/docs
EOF
}

# Call with -h or --help
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_help
    exit 0
fi
```

### 주석에서 도움말 추출

```bash
#!/bin/bash

### NAME
###     myscript - Does something useful
###
### SYNOPSIS
###     myscript [OPTIONS] FILE...
###
### DESCRIPTION
###     This script processes files in various ways.
###
### OPTIONS
###     -v, --verbose    Verbose output
###     -h, --help       Show this help

show_help() {
    sed -n 's/^### \?//p' "$0"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_help
    exit 0
fi

echo "Script running..."
```

### 버전 정보

```bash
#!/bin/bash

SCRIPT_NAME=$(basename "$0")
VERSION="1.2.3"
AUTHOR="John Doe"
COPYRIGHT="Copyright (c) 2024"
LICENSE="MIT License"

show_version() {
    cat << EOF
$SCRIPT_NAME version $VERSION
$COPYRIGHT $AUTHOR

License: $LICENSE
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Written by $AUTHOR
EOF
}

if [ "${1:-}" = "--version" ] || [ "${1:-}" = "-V" ]; then
    show_version
    exit 0
fi
```

### 동적 도움말 생성

```bash
#!/bin/bash

# Define options structure
declare -A OPTIONS_HELP=(
    ["-v|--verbose"]="Enable verbose output"
    ["-o|--output FILE"]="Specify output file"
    ["-n|--count NUM"]="Number of iterations"
    ["-h|--help"]="Show this help message"
)

generate_help() {
    echo "Usage: $(basename "$0") [OPTIONS] FILE..."
    echo ""
    echo "OPTIONS:"

    for key in $(echo "${!OPTIONS_HELP[@]}" | tr ' ' '\n' | sort); do
        printf "    %-25s %s\n" "$key" "${OPTIONS_HELP[$key]}"
    done
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    generate_help
    exit 0
fi
```

## 5. 컬러 출력

색상은 CLI 출력의 가독성을 향상시킵니다.

### ANSI 컬러 코드

```bash
#!/bin/bash

# Standard colors
BLACK='\033[0;30m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'

# Bold colors
BOLD_BLACK='\033[1;30m'
BOLD_RED='\033[1;31m'
BOLD_GREEN='\033[1;32m'
BOLD_YELLOW='\033[1;33m'
BOLD_BLUE='\033[1;34m'
BOLD_MAGENTA='\033[1;35m'
BOLD_CYAN='\033[1;36m'
BOLD_WHITE='\033[1;37m'

# Background colors
BG_BLACK='\033[40m'
BG_RED='\033[41m'
BG_GREEN='\033[42m'
BG_YELLOW='\033[43m'
BG_BLUE='\033[44m'
BG_MAGENTA='\033[45m'
BG_CYAN='\033[46m'
BG_WHITE='\033[47m'

# Text styles
BOLD='\033[1m'
DIM='\033[2m'
UNDERLINE='\033[4m'
BLINK='\033[5m'
REVERSE='\033[7m'
HIDDEN='\033[8m'

# Reset
NC='\033[0m'  # No Color

# Usage
echo -e "${RED}Error message${NC}"
echo -e "${GREEN}Success message${NC}"
echo -e "${YELLOW}Warning message${NC}"
echo -e "${BLUE}Info message${NC}"
echo -e "${BOLD}${WHITE}Important${NC}"
echo -e "${UNDERLINE}Underlined text${NC}"
echo -e "${BG_RED}${WHITE}Alert${NC}"
```

### 완전한 컬러 테이블

| 코드 | 색상 | 굵게 코드 | 굵은 색상 |
|------|-------|-----------|------------|
| `\033[0;30m` | 검정 | `\033[1;30m` | 굵은 검정 |
| `\033[0;31m` | 빨강 | `\033[1;31m` | 굵은 빨강 |
| `\033[0;32m` | 초록 | `\033[1;32m` | 굵은 초록 |
| `\033[0;33m` | 노랑 | `\033[1;33m` | 굵은 노랑 |
| `\033[0;34m` | 파랑 | `\033[1;34m` | 굵은 파랑 |
| `\033[0;35m` | 마젠타 | `\033[1;35m` | 굵은 마젠타 |
| `\033[0;36m` | 시안 | `\033[1;36m` | 굵은 시안 |
| `\033[0;37m` | 흰색 | `\033[1;37m` | 굵은 흰색 |

### tput 명령어

```bash
#!/bin/bash

# Using tput (more portable)
tput_setup() {
    # Check if terminal supports colors
    if [ -t 1 ] && [ $(tput colors) -ge 8 ]; then
        RED=$(tput setaf 1)
        GREEN=$(tput setaf 2)
        YELLOW=$(tput setaf 3)
        BLUE=$(tput setaf 4)
        MAGENTA=$(tput setaf 5)
        CYAN=$(tput setaf 6)
        WHITE=$(tput setaf 7)

        BOLD=$(tput bold)
        UNDERLINE=$(tput smul)
        RESET=$(tput sgr0)
    else
        RED=""
        GREEN=""
        YELLOW=""
        BLUE=""
        MAGENTA=""
        CYAN=""
        WHITE=""
        BOLD=""
        UNDERLINE=""
        RESET=""
    fi
}

tput_setup

echo "${RED}Red text${RESET}"
echo "${GREEN}Green text${RESET}"
echo "${BOLD}${YELLOW}Bold yellow${RESET}"
```

### 조건부 컬러링

```bash
#!/bin/bash

# Detect if output is to terminal
if [ -t 1 ]; then
    # Terminal detected, use colors
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    NC='\033[0m'
else
    # Not a terminal (pipe, file, etc.), no colors
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

# Respect NO_COLOR environment variable
if [ -n "${NO_COLOR:-}" ]; then
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

echo -e "${GREEN}This is green in terminal${NC}"
echo -e "${RED}This is red in terminal${NC}"

# Test: ./script.sh             (colored)
#       ./script.sh | cat        (not colored)
#       NO_COLOR=1 ./script.sh   (not colored)
```

### 컬러 헬퍼 함수

```bash
#!/bin/bash

# Setup colors
setup_colors() {
    if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
        RED=$(tput setaf 1)
        GREEN=$(tput setaf 2)
        YELLOW=$(tput setaf 3)
        BLUE=$(tput setaf 4)
        BOLD=$(tput bold)
        RESET=$(tput sgr0)
    else
        RED="" GREEN="" YELLOW="" BLUE="" BOLD="" RESET=""
    fi
}

# Helper functions
error() {
    echo "${RED}ERROR: $*${RESET}" >&2
}

success() {
    echo "${GREEN}SUCCESS: $*${RESET}"
}

warning() {
    echo "${YELLOW}WARNING: $*${RESET}" >&2
}

info() {
    echo "${BLUE}INFO: $*${RESET}"
}

bold() {
    echo "${BOLD}$*${RESET}"
}

setup_colors

# Usage
error "Something went wrong"
success "Operation completed"
warning "This might be a problem"
info "FYI: Some information"
bold "Important message"
```

## 6. 진행 표시기

장시간 실행되는 작업의 진행 상황을 표시합니다.

### 스피너 애니메이션

```bash
#!/bin/bash

# Spinner animation
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'

    while kill -0 "$pid" 2>/dev/null; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Usage
(sleep 5) &
echo -n "Processing..."
spinner $!
echo "Done!"

# Alternative spinner with more frames
spinner_fancy() {
    local pid=$1
    local delay=0.1
    local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')

    while kill -0 "$pid" 2>/dev/null; do
        for frame in "${frames[@]}"; do
            printf "\r%s Processing..." "$frame"
            sleep $delay
            if ! kill -0 "$pid" 2>/dev/null; then
                break 2
            fi
        done
    done
    printf "\r✓ Done!       \n"
}

# Test fancy spinner
(sleep 3) &
spinner_fancy $!
```

### 진행 막대

```bash
#!/bin/bash

# Progress bar
progress_bar() {
    local current=$1
    local total=$2
    local width=${3:-50}

    local percent=$((current * 100 / total))
    local completed=$((width * current / total))
    local remaining=$((width - completed))

    printf "\r["
    printf "%${completed}s" | tr ' ' '='
    printf "%${remaining}s" | tr ' ' ' '
    printf "] %3d%%" "$percent"

    if [ "$current" -eq "$total" ]; then
        echo ""
    fi
}

# Usage
total=100
for i in $(seq 1 $total); do
    progress_bar $i $total
    sleep 0.05
done

# Percentage-based progress
show_progress() {
    local percent=$1
    local width=50
    local completed=$((width * percent / 100))
    local remaining=$((width - completed))

    printf "\rProgress: ["
    printf "%${completed}s" | tr ' ' '█'
    printf "%${remaining}s" | tr ' ' '░'
    printf "] %3d%%" "$percent"
}

# Test
for i in $(seq 0 5 100); do
    show_progress $i
    sleep 0.2
done
echo ""
```

### 파일 다운로드 진행 상황

```bash
#!/bin/bash

# Simulate file download with progress
download_with_progress() {
    local url=$1
    local output=$2
    local total_size=${3:-1000000}  # Bytes

    echo "Downloading: $url"

    local downloaded=0
    local chunk_size=10000

    while [ $downloaded -lt $total_size ]; do
        # Simulate download
        sleep 0.1
        downloaded=$((downloaded + chunk_size))

        if [ $downloaded -gt $total_size ]; then
            downloaded=$total_size
        fi

        # Calculate progress
        local percent=$((downloaded * 100 / total_size))
        local mb_downloaded=$((downloaded / 1024 / 1024))
        local mb_total=$((total_size / 1024 / 1024))

        # Show progress
        printf "\r[%-50s] %d%% (%dMB/%dMB)" \
            $(printf '%*s' $((percent / 2)) | tr ' ' '=') \
            "$percent" \
            "$mb_downloaded" \
            "$mb_total"
    done

    echo ""
    echo "Download complete: $output"
}

# Test
download_with_progress "https://example.com/file.zip" "file.zip" 5000000
```

### 다중 라인 진행 표시

```bash
#!/bin/bash

# Multi-line progress (useful for parallel tasks)
show_multi_progress() {
    local -n tasks=$1

    # Save cursor position
    tput sc

    while true; do
        local all_done=1

        # Restore cursor position
        tput rc

        for i in "${!tasks[@]}"; do
            local task="${tasks[$i]}"
            local status=$(get_task_status "$task")
            local percent=$(get_task_percent "$task")

            printf "Task %d: [%-30s] %3d%%\n" \
                "$i" \
                "$(printf '%*s' $((percent * 30 / 100)) | tr ' ' '=')" \
                "$percent"

            if [ "$percent" -lt 100 ]; then
                all_done=0
            fi
        done

        [ $all_done -eq 1 ] && break
        sleep 0.5
    done
}

# Simpler version for demonstration
demo_multi_progress() {
    local tasks=("Task 1" "Task 2" "Task 3")
    local progress=(0 0 0)

    while true; do
        clear
        echo "=== Progress Dashboard ==="
        echo ""

        local all_done=1
        for i in "${!tasks[@]}"; do
            printf "%s: [%-30s] %3d%%\n" \
                "${tasks[$i]}" \
                "$(printf '%*s' $((progress[$i] * 30 / 100)) | tr ' ' '#')" \
                "${progress[$i]}"

            if [ ${progress[$i]} -lt 100 ]; then
                all_done=0
                progress[$i]=$((progress[$i] + RANDOM % 20))
                if [ ${progress[$i]} -gt 100 ]; then
                    progress[$i]=100
                fi
            fi
        done

        [ $all_done -eq 1 ] && break
        sleep 0.5
    done

    echo ""
    echo "All tasks completed!"
}

demo_multi_progress
```

## 7. 대화형 입력

사용자 입력을 효과적으로 수집합니다.

### 기본 입력

```bash
#!/bin/bash

# Simple input
read -p "Enter your name: " name
echo "Hello, $name!"

# Input with default value
read -p "Enter filename [default.txt]: " filename
filename=${filename:-default.txt}
echo "Using: $filename"

# Input with timeout
if read -t 5 -p "Enter something (5s timeout): " input; then
    echo "You entered: $input"
else
    echo -e "\nTimeout!"
fi
```

### 비밀번호 입력

```bash
#!/bin/bash

# Hidden input (for passwords)
read -sp "Enter password: " password
echo ""
echo "Password length: ${#password}"

# Password with confirmation
read_password() {
    local password
    local password_confirm

    while true; do
        read -sp "Enter password: " password
        echo ""

        read -sp "Confirm password: " password_confirm
        echo ""

        if [ "$password" = "$password_confirm" ]; then
            echo "$password"
            return 0
        else
            echo "Passwords don't match. Try again."
        fi
    done
}

# Usage
user_password=$(read_password)
echo "Password set successfully"
```

### 예/아니오 확인

```bash
#!/bin/bash

# Simple yes/no
ask_yes_no() {
    local prompt=$1
    local default=${2:-}

    if [ "$default" = "y" ]; then
        prompt="$prompt [Y/n]: "
    elif [ "$default" = "n" ]; then
        prompt="$prompt [y/N]: "
    else
        prompt="$prompt [y/n]: "
    fi

    while true; do
        read -p "$prompt" response

        # Use default if no response
        if [ -z "$response" ] && [ -n "$default" ]; then
            response=$default
        fi

        case "$response" in
            [Yy]*) return 0 ;;
            [Nn]*) return 1 ;;
            *) echo "Please answer yes or no." ;;
        esac
    done
}

# Usage
if ask_yes_no "Do you want to continue?" "y"; then
    echo "Continuing..."
else
    echo "Aborted"
    exit 1
fi
```

### 메뉴 선택

```bash
#!/bin/bash

# Menu selection
show_menu() {
    local prompt=$1
    shift
    local options=("$@")

    echo "$prompt"
    echo ""

    for i in "${!options[@]}"; do
        echo "  $((i + 1)). ${options[$i]}"
    done

    echo ""

    while true; do
        read -p "Enter choice [1-${#options[@]}]: " choice

        if [[ "$choice" =~ ^[0-9]+$ ]] && \
           [ "$choice" -ge 1 ] && \
           [ "$choice" -le "${#options[@]}" ]; then
            echo "$((choice - 1))"
            return 0
        else
            echo "Invalid choice. Please try again."
        fi
    done
}

# Usage
options=("Option A" "Option B" "Option C" "Quit")
selected=$(show_menu "Please select an option:" "${options[@]}")

echo "You selected: ${options[$selected]}"
```

### 검증을 포함한 고급 입력

```bash
#!/bin/bash

# Input with validation
read_validated() {
    local prompt=$1
    local validator=$2
    local error_msg=$3

    while true; do
        read -p "$prompt" input

        if eval "$validator"; then
            echo "$input"
            return 0
        else
            echo "$error_msg" >&2
        fi
    done
}

# Validators
is_number() { [[ "$input" =~ ^[0-9]+$ ]]; }
is_email() { [[ "$input" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; }
is_not_empty() { [ -n "$input" ]; }

# Usage
name=$(read_validated "Enter your name: " "is_not_empty" "Name cannot be empty")
age=$(read_validated "Enter your age: " "is_number" "Age must be a number")
email=$(read_validated "Enter email: " "is_email" "Invalid email format")

echo "Name: $name"
echo "Age: $age"
echo "Email: $email"
```

## 8. 완전한 CLI 도구 예제

모든 것을 전문적인 CLI 도구로 통합합니다.

```bash
#!/bin/bash

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_NAME=$(basename "$0")
VERSION="1.0.0"
AUTHOR="Your Name"

# Colors
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    BOLD=$(tput bold)
    RESET=$(tput sgr0)
else
    RED="" GREEN="" YELLOW="" BLUE="" BOLD="" RESET=""
fi

# Default values
VERBOSE=0
DRY_RUN=0
OUTPUT_FILE=""
LOG_FILE="/tmp/${SCRIPT_NAME}.log"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

error() { echo "${RED}ERROR: $*${RESET}" >&2; }
success() { echo "${GREEN}SUCCESS: $*${RESET}"; }
warning() { echo "${YELLOW}WARNING: $*${RESET}" >&2; }
info() { echo "${BLUE}INFO: $*${RESET}"; }
verbose() { [ $VERBOSE -eq 1 ] && echo "${BLUE}VERBOSE: $*${RESET}"; }

die() {
    local code=$1
    shift
    error "$*"
    exit "$code"
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

progress_bar() {
    local current=$1
    local total=$2
    local width=40

    local percent=$((current * 100 / total))
    local completed=$((width * current / total))

    printf "\r${BLUE}Progress:${RESET} ["
    printf "%${completed}s" | tr ' ' '='
    printf "%$((width - completed))s" | tr ' ' ' '
    printf "] %3d%%" "$percent"

    [ "$current" -eq "$total" ] && echo ""
}

# ============================================================================
# USAGE AND VERSION
# ============================================================================

show_version() {
    cat << EOF
$SCRIPT_NAME version $VERSION
Written by $AUTHOR
EOF
    exit 0
}

show_help() {
    cat << EOF
${BOLD}NAME${RESET}
    $SCRIPT_NAME - Process files with various options

${BOLD}SYNOPSIS${RESET}
    $SCRIPT_NAME [OPTIONS] FILE...

${BOLD}DESCRIPTION${RESET}
    This tool processes files with configurable options.
    It demonstrates best practices for CLI argument parsing.

${BOLD}OPTIONS${RESET}
    -v, --verbose
        Enable verbose output

    -n, --dry-run
        Perform a dry run without making changes

    -o, --output FILE
        Specify output file (default: stdout)

    -l, --log FILE
        Specify log file (default: /tmp/$SCRIPT_NAME.log)

    -h, --help
        Show this help message

    -V, --version
        Show version information

${BOLD}EXAMPLES${RESET}
    # Basic usage
    $SCRIPT_NAME file1.txt file2.txt

    # Verbose mode with output file
    $SCRIPT_NAME -v --output=result.txt input.txt

    # Dry run
    $SCRIPT_NAME -n *.txt

${BOLD}EXIT STATUS${RESET}
    0   Success
    1   General error
    2   Invalid arguments

${BOLD}AUTHOR${RESET}
    Written by $AUTHOR
EOF
    exit 0
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parse_arguments() {
    local files=()

    while [ $# -gt 0 ]; do
        case "$1" in
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=1
                shift
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --output=*)
                OUTPUT_FILE="${1#*=}"
                shift
                ;;
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            --log=*)
                LOG_FILE="${1#*=}"
                shift
                ;;
            -h|--help)
                show_help
                ;;
            -V|--version)
                show_version
                ;;
            --)
                shift
                files+=("$@")
                break
                ;;
            -*)
                die 2 "Unknown option: $1\nRun '$SCRIPT_NAME --help' for usage"
                ;;
            *)
                files+=("$1")
                shift
                ;;
        esac
    done

    # Validate
    if [ ${#files[@]} -eq 0 ]; then
        die 2 "No input files specified\nRun '$SCRIPT_NAME --help' for usage"
    fi

    # Check files exist
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            die 1 "File not found: $file"
        fi
    done

    echo "${files[@]}"
}

# ============================================================================
# MAIN LOGIC
# ============================================================================

process_file() {
    local file=$1

    verbose "Processing file: $file"
    log "Processing: $file"

    # Simulate work
    sleep 0.5

    verbose "Completed: $file"
    log "Completed: $file"
}

main() {
    log "Script started"
    verbose "Verbose mode enabled"
    [ $DRY_RUN -eq 1 ] && warning "DRY RUN MODE"

    # Parse arguments
    local files
    IFS=' ' read -ra files <<< "$(parse_arguments "$@")"

    info "Processing ${#files[@]} file(s)..."

    # Process files
    local count=0
    local total=${#files[@]}

    for file in "${files[@]}"; do
        ((count++))
        progress_bar $count $total

        if [ $DRY_RUN -eq 0 ]; then
            process_file "$file"
        fi
    done

    success "All files processed successfully"
    log "Script completed"

    # Save output
    if [ -n "$OUTPUT_FILE" ]; then
        echo "Results saved to: $OUTPUT_FILE" > "$OUTPUT_FILE"
        info "Output saved to: $OUTPUT_FILE"
    fi
}

# ============================================================================
# ENTRY POINT
# ============================================================================

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
```

## 9. 연습 문제

### 문제 1: 고급 옵션 파서

다음을 지원하는 유연한 옵션 파서를 생성하세요:
- 짧은 옵션(-v, -o file)
- 긴 옵션(--verbose, --output=file)
- 결합된 짧은 옵션(-vxf)
- 선택적 vs 필수 옵션 인수
- 불린 플래그와 값 옵션
- 위치 인수
- `--` 구분자
- 각 옵션 유형에 대한 검증
- 옵션 정의에서 자동 생성된 도움말

### 문제 2: 설정 파일 통합

다음을 수행하는 CLI 도구를 구축하세요:
- 명령줄, 설정 파일, 환경 변수에서 옵션 받기
- 우선순위 사용: CLI > 환경 > 설정 파일 > 기본값
- 여러 설정 파일 형식 지원(INI, JSON, YAML)
- 모든 설정 값 검증
- 현재 유효한 설정 출력 가능
- 설정 파일 경로를 지정하는 `--config` 옵션 포함

### 문제 3: 대화형 설정 마법사

다음을 수행하는 대화형 설정 마법사를 생성하세요:
- 설정을 통해 사용자 안내
- 각 입력 검증
- 다중 선택 옵션에 대한 메뉴 표시
- 이전 단계로 돌아갈 수 있음
- 저장 전 확인
- 설정 파일 생성
- 대화형 및 비대화형 모드 모두 보유(자동화용)
- 컬러 출력 및 진행 표시기 포함

### 문제 4: Git 스타일 하위 명령 인터페이스

git 스타일 하위 명령을 가진 CLI 도구를 구현하세요:
- 메인 명령: `mytool <subcommand> [options]`
- 여러 하위 명령(init, add, remove, list 등)
- 각 하위 명령은 자체 옵션과 도움말을 가짐
- 공유 전역 옵션(--verbose, --config)
- 탭 완성 지원(bash-completion 스크립트)
- 도움말 텍스트에서 맨 페이지 생성
- 모든 하위 명령에 걸친 일관된 에러 처리

### 문제 5: CLI 대시보드

다음을 수행하는 대화형 CLI 대시보드를 구축하세요:
- 여러 프로세스의 실시간 상태 표시
- 스크롤 없이 매초 디스플레이 업데이트
- 시각적 매력을 위해 색상 및 유니코드 문자 사용
- 키보드 명령 받기(q=종료, r=새로고침, p=일시정지)
- 실행 중인 작업의 진행 막대 표시
- 모든 이벤트를 파일에 로그
- 스크립트용 비대화형 모드에서 실행 가능

## 연습 문제

### 연습 1: 수동 파싱 vs getopts 비교

동일한 CLI 도구를 두 번 작성하세요 — 한 번은 수동 `while/case` 루프를 사용하고, 한 번은 `getopts`를 사용합니다. 도구는 다음을 받아야 합니다:
- `-v` / `--verbose` (플래그)
- `-o FILE` / `--output FILE` (인수를 갖는 옵션)
- `-n NUM` / `--count NUM` (기본값 1인 숫자 인수 옵션)
- 나머지 위치 인수(positional arguments)를 배열에 수집

두 버전을 구현한 후, 각 방식의 장점 하나와 한계 하나를 나열하세요.

### 연습 2: 자기 문서화(Self-Documenting) 도움말 함수 작성하기

`--help`, `--source DIR`, `--dest DIR`, `--compress`, `--dry-run` 플래그를 받는 스크립트 `backup.sh`를 만드세요. NAME / SYNOPSIS / DESCRIPTION / OPTIONS 규약을 따르는 `usage()` 함수를 구현하세요:

```
NAME
    backup.sh - archive files from source to destination

SYNOPSIS
    backup.sh [OPTIONS]

OPTIONS
    -s, --source DIR    Source directory to backup
    -d, --dest DIR      Destination directory
    -c, --compress      Compress the archive with gzip
    -n, --dry-run       Show what would be done without doing it
    -h, --help          Show this help and exit
```

`--help`가 전달되거나 필수 인수가 누락될 때 `usage()`를 호출하세요.

### 연습 3: NO_COLOR 지원을 포함한 컬러 출력 추가하기

기존 스크립트에 컬러 출력 라이브러리를 추가하세요. 요구사항:
- ANSI 이스케이프 코드(ANSI escape code)를 사용하여 `RED`, `GREEN`, `YELLOW`, `CYAN`, `RESET` 상수 정의
- `[ -t 1 ]`(stdout이 터미널인지)과 `${NO_COLOR:-}`(NO_COLOR 규약) 모두 확인하여 컬러 출력 여부 결정
- 적절한 컬러를 사용하는 `print_success`, `print_warn`, `print_error` 헬퍼 함수 제공
- 다음으로 테스트: `./script.sh`(컬러), `NO_COLOR=1 ./script.sh`(컬러 없음), `./script.sh | cat`(파이프, 컬러 없음)

### 연습 4: 진행 막대(Progress Bar) 구축하기

다음을 수행하는 `progress_bar <current> <total> <label>` 함수를 구현하세요:
- 완료 백분율 계산
- 40열로 스케일된 `#` 문자를 사용하여 막대 그리기
- `\r`(캐리지 리턴)을 사용하여 이전 줄을 덮어쓰면서 막대가 제자리에서 업데이트되도록
- 막대 뒤에 표시되는 선택적 레이블 문자열 받기

파일 처리를 시뮬레이션하는 루프로 테스트:

```bash
total=20
for i in $(seq 1 $total); do
    sleep 0.1
    progress_bar "$i" "$total" "Processing files"
done
echo ""  # newline after bar completes
```

### 연습 5: 다단계 대화형(Interactive) 위저드 구축하기

대화형 프롬프트를 통해 설정을 수집하는 `wizard.sh`를 구현하세요:
- 프로젝트 이름 요청(검증: 비어 있지 않아야 하고, 영숫자 + 밑줄만 허용)
- 포트 번호 요청(검증: 1024~65535 사이의 정수)
- 번호 메뉴에서 환경(environment) 선택: `1) development  2) staging  3) production`
- 기본값을 "n"으로 설정하여 계속 진행 전 확인 요청
- 비대화형 모드(stdin이 터미널이 아닌 경우), 프롬프트 대신 환경 변수 `PROJ_NAME`, `PROJ_PORT`, `PROJ_ENV`에서 값 읽기

---

**이전**: [10_Error_Handling.md](./10_Error_Handling.md) | **다음**: [12_Portability_and_Best_Practices.md](./12_Portability_and_Best_Practices.md)
