#!/bin/bash
# Exercises for Lesson 01: Shell Fundamentals and Execution Environment
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Shell Detection Script ===
# Problem: Write a script that detects which shell is running, whether it's
# login or non-login, interactive or non-interactive, and the shell version.
exercise_1() {
    echo "=== Exercise 1: Shell Detection Script ==="

    # Detect which shell is running
    if [ -n "$BASH_VERSION" ]; then
        shell_name="bash"
        shell_version="$BASH_VERSION"
    elif [ -n "$ZSH_VERSION" ]; then
        shell_name="zsh"
        shell_version="$ZSH_VERSION"
    elif [ -n "$KSH_VERSION" ]; then
        shell_name="ksh"
        shell_version="$KSH_VERSION"
    else
        shell_name="sh (unknown)"
        shell_version="unknown"
    fi

    # Detect login vs non-login shell
    login_status="non-login"
    if shopt -q login_shell 2>/dev/null; then
        login_status="login"
    fi

    # Detect interactive vs non-interactive
    interactive_status="non-interactive"
    case "$-" in
        *i*) interactive_status="interactive" ;;
    esac

    # Output results
    echo "Shell: $shell_name"
    echo "Version: $shell_version"
    echo "Type: $login_status, $interactive_status"
    echo "Shell flags (\$-): $-"
    echo "PID: $$"
    echo "PPID: $PPID"

    # Also check if stdin is a terminal
    if [ -t 0 ]; then
        echo "stdin: terminal"
    else
        echo "stdin: not a terminal (piped/redirected)"
    fi
}

# === Exercise 2: Startup File Analyzer ===
# Problem: List all bash startup files, their load order, and display first
# 5 lines of each. Check for common mistakes.
exercise_2() {
    echo "=== Exercise 2: Startup File Analyzer ==="

    # Define startup files in load order
    local login_files=(
        "/etc/profile"
        "$HOME/.bash_profile"
        "$HOME/.bash_login"
        "$HOME/.profile"
    )

    local interactive_files=(
        "/etc/bash.bashrc"
        "$HOME/.bashrc"
    )

    local logout_files=(
        "$HOME/.bash_logout"
    )

    echo "--- Login Shell Startup Files (in load order) ---"
    for file in "${login_files[@]}"; do
        if [ -f "$file" ]; then
            echo "[EXISTS] $file"
            echo "  First 5 lines:"
            head -5 "$file" 2>/dev/null | while IFS= read -r line; do
                echo "    $line"
            done
        else
            echo "[MISSING] $file"
        fi
        echo ""
    done

    echo "--- Non-Login Interactive Shell Files ---"
    for file in "${interactive_files[@]}"; do
        if [ -f "$file" ]; then
            echo "[EXISTS] $file"
            echo "  First 5 lines:"
            head -5 "$file" 2>/dev/null | while IFS= read -r line; do
                echo "    $line"
            done
        else
            echo "[MISSING] $file"
        fi
        echo ""
    done

    echo "--- Logout Files ---"
    for file in "${logout_files[@]}"; do
        if [ -f "$file" ]; then
            echo "[EXISTS] $file"
        else
            echo "[MISSING] $file"
        fi
    done

    # Check for common mistakes
    echo ""
    echo "--- Common Mistake Checks ---"

    # Check if .bash_profile sources .bashrc
    if [ -f "$HOME/.bash_profile" ]; then
        if grep -q "\.bashrc" "$HOME/.bash_profile" 2>/dev/null; then
            echo "[OK] .bash_profile sources .bashrc"
        else
            echo "[WARN] .bash_profile does NOT source .bashrc"
            echo "  Tip: Add 'source ~/.bashrc' to ~/.bash_profile"
        fi
    fi

    # Check if aliases are in .bash_profile instead of .bashrc
    if [ -f "$HOME/.bash_profile" ]; then
        if grep -q "^alias " "$HOME/.bash_profile" 2>/dev/null; then
            echo "[WARN] Aliases found in .bash_profile instead of .bashrc"
            echo "  Tip: Move aliases to .bashrc for non-login shell support"
        else
            echo "[OK] No aliases in .bash_profile"
        fi
    fi

    # Check if .bashrc has an interactive guard
    if [ -f "$HOME/.bashrc" ]; then
        if grep -q 'case \$-' "$HOME/.bashrc" 2>/dev/null || \
           grep -q '\[\[ \$- ==' "$HOME/.bashrc" 2>/dev/null; then
            echo "[OK] .bashrc has interactive shell guard"
        else
            echo "[WARN] .bashrc may lack an interactive shell guard"
        fi
    fi
}

# === Exercise 3: Exit Code Logger ===
# Problem: Write a function that wraps any command, logs the command,
# exit code, and execution time to a log file.
exercise_3() {
    echo "=== Exercise 3: Exit Code Logger ==="

    local log_file="/tmp/command_log_$$.txt"

    # The log_command wrapper function
    log_command() {
        local cmd_string="$*"
        local start_time
        local end_time
        local duration
        local exit_code

        # Record start time (seconds since epoch with fractional part)
        start_time=$(date +%s%N 2>/dev/null || date +%s)

        # Execute the command
        "$@"
        exit_code=$?

        # Record end time
        end_time=$(date +%s%N 2>/dev/null || date +%s)

        # Calculate duration in seconds
        if [[ "$start_time" =~ ^[0-9]+$ ]] && (( start_time > 1000000000000 )); then
            # Nanosecond precision available
            duration=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc 2>/dev/null || echo "N/A")
        else
            duration=$(( end_time - start_time ))
        fi

        # Format timestamp
        local timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')

        # Log to file
        echo "$timestamp | $cmd_string | $exit_code | ${duration}s" >> "$log_file"

        # Also print to stdout
        echo "  Logged: $timestamp | $cmd_string | exit=$exit_code | ${duration}s"

        return $exit_code
    }

    # Test with various commands
    echo "Running test commands..."
    log_command echo "Hello World"
    log_command ls /tmp > /dev/null
    log_command ls /nonexistent_directory 2>/dev/null
    log_command sleep 0.1
    log_command true
    log_command false

    echo ""
    echo "--- Log File Contents ($log_file) ---"
    cat "$log_file"

    # Cleanup
    rm -f "$log_file"
}

# === Exercise 4: Portable Script Checker ===
# Problem: Analyze a bash script and report non-POSIX constructs,
# bashisms, and provide a portability score.
exercise_4() {
    echo "=== Exercise 4: Portable Script Checker ==="

    # Create a sample script to analyze
    local test_script="/tmp/test_script_$$.sh"
    cat > "$test_script" << 'SCRIPT'
#!/bin/bash
declare -A config
config[host]="localhost"

if [[ $1 == "start" ]]; then
    echo "Starting..."
fi

function cleanup {
    local temp=$1
    rm -f "$temp"
}

array=(one two three)
for item in "${array[@]}"; do
    echo "$item"
done

result=$(( 5 + 3 ))
echo "Result: $result"

cat <<< "Here string"

if [ -f /etc/passwd ]; then
    echo "Found"
fi

case "$1" in
    start) echo "Starting" ;;
    stop) echo "Stopping" ;;
esac
SCRIPT

    check_portability() {
        local script_file="$1"
        local total_checks=0
        local issues=0

        echo "Analyzing: $script_file"
        echo ""

        # Check for [[ ]]
        local count
        count=$(grep -c '\[\[' "$script_file" 2>/dev/null || echo 0)
        (( total_checks++ ))
        if (( count > 0 )); then
            echo "[BASHISM] [[ ]] used $count time(s) - Use [ ] for POSIX"
            (( issues++ ))
        fi

        # Check for declare
        count=$(grep -c 'declare' "$script_file" 2>/dev/null || echo 0)
        (( total_checks++ ))
        if (( count > 0 )); then
            echo "[BASHISM] 'declare' used $count time(s) - Not in POSIX"
            (( issues++ ))
        fi

        # Check for arrays
        count=$(grep -c '=(' "$script_file" 2>/dev/null || echo 0)
        (( total_checks++ ))
        if (( count > 0 )); then
            echo "[BASHISM] Arrays used $count time(s) - Not in POSIX"
            (( issues++ ))
        fi

        # Check for function keyword
        count=$(grep -c '^function ' "$script_file" 2>/dev/null || echo 0)
        (( total_checks++ ))
        if (( count > 0 )); then
            echo "[BASHISM] 'function' keyword used $count time(s) - Use name() { } for POSIX"
            (( issues++ ))
        fi

        # Check for local keyword
        count=$(grep -c 'local ' "$script_file" 2>/dev/null || echo 0)
        (( total_checks++ ))
        if (( count > 0 )); then
            echo "[WARNING] 'local' used $count time(s) - Widely supported but not POSIX"
            (( issues++ ))
        fi

        # Check for here strings <<<
        count=$(grep -c '<<<' "$script_file" 2>/dev/null || echo 0)
        (( total_checks++ ))
        if (( count > 0 )); then
            echo "[BASHISM] Here strings (<<<) used $count time(s) - Not in POSIX"
            (( issues++ ))
        fi

        # Check for process substitution <()
        count=$(grep -c '<(' "$script_file" 2>/dev/null || echo 0)
        (( total_checks++ ))
        if (( count > 0 )); then
            echo "[BASHISM] Process substitution <() used $count time(s)"
            (( issues++ ))
        fi

        # Check for #!/bin/bash vs #!/bin/sh
        if head -1 "$script_file" | grep -q '#!/bin/bash'; then
            echo "[INFO] Shebang: #!/bin/bash (explicitly targeting bash)"
            (( total_checks++ ))
        fi

        # Calculate portability score
        echo ""
        local score
        if (( total_checks > 0 )); then
            score=$(( (total_checks - issues) * 100 / total_checks ))
        else
            score=100
        fi

        echo "--- Summary ---"
        echo "Checks performed: $total_checks"
        echo "Issues found: $issues"
        echo "Portability score: ${score}%"

        if (( score >= 80 )); then
            echo "Rating: Good portability"
        elif (( score >= 50 )); then
            echo "Rating: Moderate - some bash-specific features"
        else
            echo "Rating: Low - heavily depends on bash features"
        fi
    }

    check_portability "$test_script"

    # Cleanup
    rm -f "$test_script"
}

# === Exercise 5: Environment Snapshot ===
# Problem: Save/restore/diff environment variables and shell options.
exercise_5() {
    echo "=== Exercise 5: Environment Snapshot ==="

    local snapshot_file="/tmp/envsnap_$$.env"
    local options_file="/tmp/envsnap_opts_$$.env"

    # Save function
    envsnap_save() {
        local snap_file="$1"
        local opts_file="${snap_file}.opts"

        echo "Saving environment snapshot to $snap_file..."

        # Save environment variables
        env | sort > "$snap_file"

        # Save shell options (set -o and shopt)
        {
            echo "# set options"
            set -o
            echo "# shopt options"
            shopt
        } > "$opts_file"

        echo "  Saved $(wc -l < "$snap_file") environment variables"
        echo "  Saved shell options to ${opts_file}"
    }

    # Diff function
    envsnap_diff() {
        local snap_file="$1"

        if [ ! -f "$snap_file" ]; then
            echo "Error: Snapshot file not found: $snap_file"
            return 1
        fi

        echo "Comparing current environment with snapshot..."
        echo ""

        # Get current environment
        local current_file="/tmp/envsnap_current_$$.env"
        env | sort > "$current_file"

        # Show differences
        local added removed changed
        added=$(diff "$snap_file" "$current_file" | grep '^>' | wc -l)
        removed=$(diff "$snap_file" "$current_file" | grep '^<' | wc -l)

        echo "Added variables: $added"
        echo "Removed variables: $removed"

        if (( added > 0 )); then
            echo ""
            echo "--- Added ---"
            diff "$snap_file" "$current_file" | grep '^>' | head -5 | sed 's/^> /  /'
        fi

        if (( removed > 0 )); then
            echo ""
            echo "--- Removed ---"
            diff "$snap_file" "$current_file" | grep '^<' | head -5 | sed 's/^< /  /'
        fi

        rm -f "$current_file"
    }

    # Restore function (simulated -- echoes commands instead of actually running them)
    envsnap_restore() {
        local snap_file="$1"

        if [ ! -f "$snap_file" ]; then
            echo "Error: Snapshot file not found: $snap_file"
            return 1
        fi

        echo "Restoring environment from snapshot (simulated)..."
        echo "  Would restore $(wc -l < "$snap_file") variables"
        echo "  (In a real script, this would source the exported variables)"
    }

    # Demonstrate the workflow
    envsnap_save "$snapshot_file"
    echo ""

    # Make a change
    export TEST_ENVSNAP_VAR="hello_from_exercise_5"
    echo "Added TEST_ENVSNAP_VAR to environment"
    echo ""

    envsnap_diff "$snapshot_file"
    echo ""

    envsnap_restore "$snapshot_file"

    # Cleanup
    unset TEST_ENVSNAP_VAR
    rm -f "$snapshot_file" "${snapshot_file}.opts"
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
