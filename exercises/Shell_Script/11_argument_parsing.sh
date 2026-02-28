#!/bin/bash
# Exercises for Lesson 11: Argument Parsing and CLI Interfaces
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Compare Manual Parsing vs getopts ===
# Problem: Write the same CLI tool twice — manual while/case and getopts.
# Options: -v/--verbose, -o FILE/--output FILE, -n NUM/--count NUM (default 1),
# plus positional arguments collected into an array.
exercise_1() {
    echo "=== Exercise 1: Compare Manual Parsing vs getopts ==="

    # --- Version A: Manual while/case loop ---
    parse_manual() {
        local verbose=0
        local output=""
        local count=1
        local positional=()

        while [ $# -gt 0 ]; do
            case "$1" in
                -v|--verbose)
                    verbose=1
                    shift
                    ;;
                -o|--output)
                    output="$2"
                    shift 2
                    ;;
                --output=*)
                    output="${1#*=}"
                    shift
                    ;;
                -n|--count)
                    count="$2"
                    shift 2
                    ;;
                --count=*)
                    count="${1#*=}"
                    shift
                    ;;
                --)
                    shift
                    positional+=("$@")
                    break
                    ;;
                -*)
                    echo "  [Manual] Unknown option: $1" >&2
                    return 1
                    ;;
                *)
                    positional+=("$1")
                    shift
                    ;;
            esac
        done

        echo "  [Manual] verbose=$verbose output='$output' count=$count"
        echo "  [Manual] positional=(${positional[*]})"
    }

    # --- Version B: getopts (short options only) ---
    parse_getopts() {
        local OPTIND opt
        local verbose=0
        local output=""
        local count=1

        while getopts ":vo:n:" opt; do
            case "$opt" in
                v)  verbose=1 ;;
                o)  output="$OPTARG" ;;
                n)  count="$OPTARG" ;;
                \?) echo "  [getopts] Unknown option: -$OPTARG" >&2; return 1 ;;
                :)  echo "  [getopts] Option -$OPTARG requires an argument" >&2; return 1 ;;
            esac
        done
        shift $((OPTIND - 1))

        echo "  [getopts] verbose=$verbose output='$output' count=$count"
        echo "  [getopts] positional=($*)"
    }

    # Test cases
    echo "--- Test 1: All options ---"
    parse_manual -v -o result.txt -n 5 file1.txt file2.txt
    parse_getopts -v -o result.txt -n 5 file1.txt file2.txt

    echo ""
    echo "--- Test 2: Long options (manual only) ---"
    parse_manual --verbose --output=result.txt --count=3 data.csv
    echo "  [getopts] (Long options not supported by getopts)"

    echo ""
    echo "--- Test 3: Defaults (no options) ---"
    parse_manual file.txt
    parse_getopts file.txt

    echo ""
    echo "--- Test 4: -- separator ---"
    parse_manual -v -- -not-an-option file.txt
    parse_getopts -v -- -not-an-option file.txt

    echo ""
    echo "--- Comparison ---"
    echo "  Manual parsing: + supports long options, --key=value, -- separator"
    echo "                  - more code, must handle edge cases manually"
    echo "  getopts:        + built-in, handles -abc bundling, POSIX portable"
    echo "                  - no long options, no --key=value syntax"
}

# === Exercise 2: Write a Self-Documenting Help Function ===
# Problem: Create backup.sh with --help, --source DIR, --dest DIR,
# --compress, --dry-run. usage() follows NAME/SYNOPSIS/OPTIONS convention.
exercise_2() {
    echo "=== Exercise 2: Write a Self-Documenting Help Function ==="

    # Simulate backup.sh entirely within this function
    local SCRIPT_NAME="backup.sh"
    local source_dir=""
    local dest_dir=""
    local compress=false
    local dry_run=false

    usage() {
        cat << EOF
NAME
    $SCRIPT_NAME - archive files from source to destination

SYNOPSIS
    $SCRIPT_NAME [OPTIONS]

DESCRIPTION
    Copies files from a source directory to a destination directory,
    optionally compressing the result with gzip.

OPTIONS
    -s, --source DIR    Source directory to backup
    -d, --dest DIR      Destination directory
    -c, --compress      Compress the archive with gzip
    -n, --dry-run       Show what would be done without doing it
    -h, --help          Show this help and exit

EXAMPLES
    $SCRIPT_NAME --source /var/data --dest /backup
    $SCRIPT_NAME -s /home/user -d /mnt/backup -c
    $SCRIPT_NAME --dry-run -s /tmp/data -d /tmp/out

EXIT STATUS
    0   Success
    1   General error
    2   Missing required arguments
EOF
    }

    parse_backup_args() {
        while [ $# -gt 0 ]; do
            case "$1" in
                -s|--source)  source_dir="$2"; shift 2 ;;
                -d|--dest)    dest_dir="$2"; shift 2 ;;
                -c|--compress) compress=true; shift ;;
                -n|--dry-run) dry_run=true; shift ;;
                -h|--help)    usage; return 0 ;;
                *)            echo "  Unknown option: $1" >&2; return 2 ;;
            esac
        done

        # Validate required arguments
        if [ -z "$source_dir" ] || [ -z "$dest_dir" ]; then
            echo "  Error: --source and --dest are required." >&2
            echo ""
            usage
            return 2
        fi

        return 0
    }

    # Test 1: --help
    echo "--- Test 1: --help output ---"
    parse_backup_args --help | sed 's/^/  /'

    # Test 2: Valid args
    echo ""
    echo "--- Test 2: Valid arguments ---"
    source_dir=""; dest_dir=""; compress=false; dry_run=false
    parse_backup_args -s /var/data -d /backup -c -n
    echo "  source=$source_dir dest=$dest_dir compress=$compress dry_run=$dry_run"

    # Test 3: Missing required args
    echo ""
    echo "--- Test 3: Missing required args ---"
    source_dir=""; dest_dir=""
    parse_backup_args -c 2>&1 | head -5 | sed 's/^/  /'
}

# === Exercise 3: Add Color Output with NO_COLOR Support ===
# Problem: Define color constants, check -t 1 and NO_COLOR, provide helper functions.
exercise_3() {
    echo "=== Exercise 3: Add Color Output with NO_COLOR Support ==="

    # Color library
    setup_colors() {
        if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
            RED='\033[0;31m'
            GREEN='\033[0;32m'
            YELLOW='\033[0;33m'
            CYAN='\033[0;36m'
            RESET='\033[0m'
            COLOR_ENABLED="yes"
        else
            RED=''
            GREEN=''
            YELLOW=''
            CYAN=''
            RESET=''
            COLOR_ENABLED="no"
        fi
    }

    print_success() { echo -e "${GREEN}SUCCESS: $*${RESET}"; }
    print_warn()    { echo -e "${YELLOW}WARNING: $*${RESET}" >&2; }
    print_error()   { echo -e "${RED}ERROR: $*${RESET}" >&2; }
    print_info()    { echo -e "${CYAN}INFO: $*${RESET}"; }

    # Test 1: Normal (color depends on terminal)
    echo "--- Test 1: setup_colors() in current terminal ---"
    setup_colors
    echo "  Color enabled: $COLOR_ENABLED"
    print_success "Build completed"
    print_warn "Disk usage high"
    print_error "Connection refused"
    print_info "Server started on port 8080"

    # Test 2: Simulate NO_COLOR=1
    echo ""
    echo "--- Test 2: Simulate NO_COLOR=1 ---"
    local saved_no_color="${NO_COLOR:-}"
    NO_COLOR=1
    setup_colors
    echo "  Color enabled: $COLOR_ENABLED"
    print_success "This should have NO color codes"
    print_error "This should also have NO color codes"
    NO_COLOR="$saved_no_color"

    # Test 3: Force colors back on
    echo ""
    echo "--- Test 3: Colors restored ---"
    unset NO_COLOR 2>/dev/null || true
    setup_colors
    echo "  Color enabled: $COLOR_ENABLED"
    print_info "Colors may be active again (if running in terminal)"

    echo ""
    echo "--- Usage notes ---"
    echo "  Run:          ./script.sh           (color if in terminal)"
    echo "  Piped:        ./script.sh | cat     (no color)"
    echo "  Explicit off: NO_COLOR=1 ./script.sh (no color)"
}

# === Exercise 4: Build a Progress Bar ===
# Problem: progress_bar <current> <total> <label> using # chars, 40 cols, \r.
exercise_4() {
    echo "=== Exercise 4: Build a Progress Bar ==="

    progress_bar() {
        local current="$1"
        local total="$2"
        local label="${3:-}"
        local width=40

        local percent=$((current * 100 / total))
        local completed=$((width * current / total))
        local remaining=$((width - completed))

        # Build the bar string
        local bar=""
        local i
        for (( i=0; i<completed; i++ )); do bar+="#"; done
        for (( i=0; i<remaining; i++ )); do bar+=" "; done

        # Print with \r to overwrite
        printf "\r  [%s] %3d%% %s" "$bar" "$percent" "$label"

        # Newline at 100%
        if [ "$current" -eq "$total" ]; then
            echo ""
        fi
    }

    echo "--- Simulating file processing ---"
    local total=20
    for i in $(seq 1 "$total"); do
        sleep 0.05
        progress_bar "$i" "$total" "Processing files"
    done

    echo ""
    echo "--- Simulating download ---"
    total=50
    for i in $(seq 1 "$total"); do
        sleep 0.02
        progress_bar "$i" "$total" "Downloading data"
    done

    echo ""
    echo "--- Simulating build ---"
    total=10
    for i in $(seq 1 "$total"); do
        sleep 0.1
        progress_bar "$i" "$total" "Building"
    done

    echo ""
    echo "  Progress bar tests complete."
}

# === Exercise 5: Build a Multi-Step Interactive Wizard ===
# Problem: wizard.sh that collects project name, port, environment via prompts.
# In non-interactive mode, reads from environment variables.
exercise_5() {
    echo "=== Exercise 5: Build a Multi-Step Interactive Wizard ==="

    # Since we're running inside a script and stdin may not be a terminal,
    # we simulate both the interactive and non-interactive modes.

    wizard() {
        local proj_name=""
        local proj_port=""
        local proj_env=""
        local is_interactive=false

        # Detect interactive mode
        if [ -t 0 ]; then
            is_interactive=true
        fi

        if $is_interactive; then
            # --- Interactive mode ---

            # Step 1: Project name
            while true; do
                read -p "  Enter project name (alphanumeric + underscores): " proj_name
                if [[ "$proj_name" =~ ^[a-zA-Z0-9_]+$ ]] && [ -n "$proj_name" ]; then
                    break
                fi
                echo "  Invalid. Must be non-empty, alphanumeric + underscores only."
            done

            # Step 2: Port number
            while true; do
                read -p "  Enter port number (1024-65535): " proj_port
                if [[ "$proj_port" =~ ^[0-9]+$ ]] && \
                   (( proj_port >= 1024 && proj_port <= 65535 )); then
                    break
                fi
                echo "  Invalid. Must be an integer between 1024 and 65535."
            done

            # Step 3: Environment
            echo "  Choose environment:"
            echo "    1) development"
            echo "    2) staging"
            echo "    3) production"
            while true; do
                read -p "  Enter choice [1-3]: " choice
                case "$choice" in
                    1) proj_env="development"; break ;;
                    2) proj_env="staging"; break ;;
                    3) proj_env="production"; break ;;
                    *) echo "  Invalid choice." ;;
                esac
            done

            # Step 4: Confirm
            echo ""
            echo "  --- Configuration Summary ---"
            echo "    Project:     $proj_name"
            echo "    Port:        $proj_port"
            echo "    Environment: $proj_env"
            echo ""
            read -p "  Proceed? [y/N]: " confirm
            if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                echo "  Aborted."
                return 1
            fi
        else
            # --- Non-interactive mode: read from environment ---
            proj_name="${PROJ_NAME:-}"
            proj_port="${PROJ_PORT:-}"
            proj_env="${PROJ_ENV:-}"

            # Validate
            if [[ ! "$proj_name" =~ ^[a-zA-Z0-9_]+$ ]] || [ -z "$proj_name" ]; then
                echo "  Error: PROJ_NAME is missing or invalid." >&2
                return 1
            fi
            if [[ ! "$proj_port" =~ ^[0-9]+$ ]] || \
               (( proj_port < 1024 || proj_port > 65535 )); then
                echo "  Error: PROJ_PORT must be 1024-65535." >&2
                return 1
            fi
            case "$proj_env" in
                development|staging|production) ;;
                *) echo "  Error: PROJ_ENV must be development/staging/production." >&2
                   return 1 ;;
            esac
        fi

        # Output the generated config
        echo "  --- Generated config.ini ---"
        echo "    [project]"
        echo "    name = $proj_name"
        echo "    port = $proj_port"
        echo "    environment = $proj_env"
        return 0
    }

    # Since stdin is typically not a terminal in exercise mode,
    # we test non-interactive mode by setting environment variables.

    echo "--- Non-interactive mode (from env vars) ---"
    echo ""
    echo "  Test 1: Valid inputs"
    (
        export PROJ_NAME="my_app"
        export PROJ_PORT="8080"
        export PROJ_ENV="staging"
        wizard
    )

    echo ""
    echo "  Test 2: Invalid project name"
    (
        export PROJ_NAME="invalid name!!"
        export PROJ_PORT="8080"
        export PROJ_ENV="production"
        wizard 2>&1 || true
    )

    echo ""
    echo "  Test 3: Invalid port"
    (
        export PROJ_NAME="valid_name"
        export PROJ_PORT="80"
        export PROJ_ENV="development"
        wizard 2>&1 || true
    )

    echo ""
    echo "  Test 4: Invalid environment"
    (
        export PROJ_NAME="valid_name"
        export PROJ_PORT="3000"
        export PROJ_ENV="invalid"
        wizard 2>&1 || true
    )
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
echo ""
echo "All exercises completed!"
