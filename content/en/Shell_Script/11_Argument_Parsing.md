# Lesson 11: Argument Parsing and CLI Interfaces

**Difficulty**: ⭐⭐⭐

**Previous**: [10_Error_Handling.md](./10_Error_Handling.md) | **Next**: [12_Portability_and_Best_Practices.md](./12_Portability_and_Best_Practices.md)

---

## 1. Manual Argument Parsing

Manual parsing gives you complete control over argument handling.

### Basic Argument Loop

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

### Handling Options with Values

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

### Advanced Manual Parsing

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

`getopts` is a POSIX built-in for parsing options.

### Basic getopts Usage

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

### getopts Error Handling

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

### getopts with Functions

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

### Complete getopts Example

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

GNU `getopt` supports long options and more advanced parsing.

### Basic getopt Usage

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

### getopt with Long Options Only

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

### getopt vs getopts Comparison

| Feature | getopts (POSIX) | getopt (GNU) |
|---------|----------------|--------------|
| Portability | POSIX (all systems) | GNU (Linux, macOS with install) |
| Long options | No | Yes |
| Option bundling | Limited | Full support |
| `--` separator | Manual handling | Built-in |
| Error messages | Basic | Detailed |
| Option reordering | No | Yes |
| Complexity | Simple | More complex |
| Use case | Simple scripts | Complex CLI tools |

### Complete getopt Example

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

## 4. Self-Documenting Help

Good help messages make CLI tools user-friendly.

### Help Message Template

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

### Extracting Help from Comments

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

### Version Information

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

### Dynamic Help Generation

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

## 5. Color Output

Colors improve readability of CLI output.

### ANSI Color Codes

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

### Complete Color Table

| Code | Color | Bold Code | Bold Color |
|------|-------|-----------|------------|
| `\033[0;30m` | Black | `\033[1;30m` | Bold Black |
| `\033[0;31m` | Red | `\033[1;31m` | Bold Red |
| `\033[0;32m` | Green | `\033[1;32m` | Bold Green |
| `\033[0;33m` | Yellow | `\033[1;33m` | Bold Yellow |
| `\033[0;34m` | Blue | `\033[1;34m` | Bold Blue |
| `\033[0;35m` | Magenta | `\033[1;35m` | Bold Magenta |
| `\033[0;36m` | Cyan | `\033[1;36m` | Bold Cyan |
| `\033[0;37m` | White | `\033[1;37m` | Bold White |

### tput Commands

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

### Conditional Coloring

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

### Color Helper Functions

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

## 6. Progress Indicators

Show progress for long-running operations.

### Spinner Animation

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

### Progress Bar

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

### File Download Progress

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

### Multi-line Progress Display

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

## 7. Interactive Input

Gathering user input effectively.

### Basic Input

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

### Password Input

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

### Yes/No Confirmation

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

### Menu Selection

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

### Advanced Input with Validation

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

## 8. Complete CLI Tool Example

Putting it all together into a professional CLI tool.

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

## 9. Practice Problems

### Problem 1: Advanced Option Parser

Create a flexible option parser that supports:
- Short options (-v, -o file)
- Long options (--verbose, --output=file)
- Combined short options (-vxf)
- Optional vs required option arguments
- Boolean flags and value options
- Positional arguments
- `--` separator
- Validation for each option type
- Auto-generated help from option definitions

### Problem 2: Configuration File Integration

Build a CLI tool that:
- Accepts options from command line, config file, and environment variables
- Uses priority: CLI > environment > config file > defaults
- Supports multiple config file formats (INI, JSON, YAML)
- Validates all configuration values
- Can output current effective configuration
- Includes `--config` option to specify config file path

### Problem 3: Interactive Setup Wizard

Create an interactive setup wizard that:
- Guides user through configuration
- Validates each input
- Shows menu for multi-choice options
- Allows going back to previous steps
- Confirms before saving
- Generates a config file
- Has both interactive and non-interactive modes (for automation)
- Includes colored output and progress indicators

### Problem 4: Git-Style Subcommand Interface

Implement a CLI tool with git-style subcommands:
- Main command: `mytool <subcommand> [options]`
- Multiple subcommands (init, add, remove, list, etc.)
- Each subcommand has its own options and help
- Shared global options (--verbose, --config)
- Tab completion support (bash-completion script)
- Man page generation from help text
- Consistent error handling across all subcommands

### Problem 5: CLI Dashboard

Build an interactive CLI dashboard that:
- Shows real-time status of multiple processes
- Updates display every second without scrolling
- Uses colors and Unicode characters for visual appeal
- Accepts keyboard commands (q=quit, r=refresh, p=pause)
- Shows progress bars for running tasks
- Logs all events to a file
- Can run in non-interactive mode for scripts

---

**Previous**: [10_Error_Handling.md](./10_Error_Handling.md) | **Next**: [12_Portability_and_Best_Practices.md](./12_Portability_and_Best_Practices.md)
