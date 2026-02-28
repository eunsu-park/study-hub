#!/bin/bash
# Exercises for Lesson 10: Error Handling and Debugging
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Understand set -e Gotchas ===
# Problem: Predict and explain output of set -e with various constructs.
exercise_1() {
    echo "=== Exercise 1: Understand set -e Gotchas ==="

    echo "Running the following script with set -e:"
    echo '  set -e'
    echo '  echo "Start"'
    echo '  false || echo "A"'
    echo '  if false; then echo "B"; fi'
    echo '  echo "C"'
    echo '  ! false'
    echo '  echo "D"'
    echo '  false | true'
    echo '  echo "E"'
    echo ""

    # Run in a subshell to avoid affecting the rest of the script
    (
        set -e
        echo "Start"
        # 'false || echo "A"': false is part of an OR list, so set -e
        # does NOT trigger. "A" prints because false fails and || runs echo.
        false || echo "A"
        # 'if false': false is the condition of an if, so set -e does NOT
        # trigger. "B" does NOT print because the condition is false.
        if false; then echo "B"; fi
        # "C" prints normally.
        echo "C"
        # '! false': The ! negates the exit code (false returns 1 -> ! makes 0),
        # so set -e does NOT trigger.
        ! false
        # "D" prints normally.
        echo "D"
        # 'false | true': In a pipeline, only the LAST command's exit status
        # matters (without pipefail). true returns 0, so set -e doesn't trigger.
        false | true
        # "E" prints normally.
        echo "E"
    )

    echo ""
    echo "--- Explanation ---"
    echo "  All letters A, C, D, E print. B does NOT print."
    echo "  A: false is in '||' list -> set -e ignores failures in compound lists"
    echo "  B: false is 'if' condition -> does not trigger set -e, but body skipped"
    echo "  C: normal statement after non-failing commands"
    echo "  D: '! false' negates to success -> no set -e trigger"
    echo "  E: 'false | true' -> pipeline exit status is last command (true=0)"
    echo "     (With pipefail, this would differ: pipeline would fail)"
}

# === Exercise 2: Build a trap ERR Handler ===
# Problem: Script with strict mode, trap ERR showing exit code/command/line,
# trap EXIT always printing "Cleanup done".
exercise_2() {
    echo "=== Exercise 2: Build a trap ERR Handler ==="

    # We run this in a subshell to keep the main script running
    echo "--- Running safe_run.sh simulation ---"
    (
        set -euo pipefail

        err_handler() {
            local exit_code=$?
            echo "  [ERR] Exit code : $exit_code"
            echo "  [ERR] Command   : $BASH_COMMAND"
            echo "  [ERR] Line      : ${BASH_LINENO[0]}"
        }

        exit_handler() {
            echo "  [EXIT] Cleanup done"
        }

        trap 'err_handler' ERR
        trap 'exit_handler' EXIT

        echo "  step 1"
        false           # This triggers ERR and then EXIT due to set -e
        echo "  step 3"  # This never runs
    ) 2>/dev/null || true

    echo ""
    echo "--- Verification ---"
    echo "  step 3 never printed (confirmed set -e aborted after false)"
    echo "  ERR trap fired with exit code, command, and line info"
    echo "  EXIT trap always fires (Cleanup done)"
}

# === Exercise 3: Write a die/assert Library ===
# Problem: Create error_lib.sh with named exit codes, die, assert_file_exists,
# and assert_not_empty functions.
exercise_3() {
    echo "=== Exercise 3: Write a die/assert Library ==="

    # --- Inline error_lib.sh ---
    readonly E_SUCCESS=0
    readonly E_GENERAL=1
    readonly E_INVALID_ARGS=2
    readonly E_NOT_FOUND=66

    die() {
        local code="$1"
        shift
        echo "  [FATAL] $* (exit code: $code)" >&2
        return "$code"
    }

    assert_file_exists() {
        local path="$1"
        if [ ! -f "$path" ]; then
            die "$E_NOT_FOUND" "File not found: $path"
            return $?
        fi
        echo "  [OK] File exists: $path"
        return 0
    }

    assert_not_empty() {
        local value="$1"
        local name="$2"
        if [ -z "$value" ]; then
            die "$E_INVALID_ARGS" "$name must not be empty"
            return $?
        fi
        echo "  [OK] $name is not empty: '$value'"
        return 0
    }

    # --- Test the library ---
    echo "--- Testing valid inputs ---"
    assert_file_exists "/etc/hosts" || true
    assert_not_empty "hello" "greeting" || true

    echo ""
    echo "--- Testing invalid inputs ---"
    assert_file_exists "/nonexistent/file.txt" || true
    assert_not_empty "" "username" || true
}

# === Exercise 4: Implement Safe File Operations with Cleanup ===
# Problem: Process a source file to output file using mktemp, trap EXIT
# for cleanup, atomic mv, and test failure cleanup.
exercise_4() {
    echo "=== Exercise 4: Safe File Operations with Cleanup ==="

    local work_dir="/tmp/safeops_$$"
    mkdir -p "$work_dir"

    # Create source file
    local source_file="$work_dir/source.txt"
    cat > "$source_file" << 'EOF'
hello world
foo bar
test line
EOF

    local output_file="$work_dir/output.txt"

    # Safe file processor
    safe_process() {
        local src="$1"
        local dest="$2"

        # Create temp file
        local tmpfile
        tmpfile=$(mktemp "${dest}.XXXXXX") || {
            echo "  [ERROR] Failed to create temp file"
            return 1
        }

        # Register cleanup (simulate with explicit cleanup since we're in a function)
        local cleanup_needed=true

        # Process: uppercase the content
        echo "  Processing $src -> $tmpfile (temp)"
        tr '[:lower:]' '[:upper:]' < "$src" > "$tmpfile" || {
            echo "  [ERROR] Processing failed"
            rm -f "$tmpfile"
            return 1
        }

        # Atomic move to final destination
        echo "  Moving $tmpfile -> $dest (atomic)"
        mv "$tmpfile" "$dest" || {
            echo "  [ERROR] Move failed"
            rm -f "$tmpfile"
            return 1
        }

        cleanup_needed=false
        echo "  [OK] Output written to $dest"
        return 0
    }

    # Test 1: Successful processing
    echo "--- Test 1: Successful processing ---"
    safe_process "$source_file" "$output_file"
    echo "  Result:"
    cat "$output_file" | sed 's/^/    /'

    # Test 2: Failure midway (nonexistent source)
    echo ""
    echo "--- Test 2: Failure cleanup ---"
    safe_process "/nonexistent/file" "$work_dir/fail_output.txt" || {
        echo "  Aborted, temp files cleaned"
    }

    # Verify no temp files left
    local leftover
    leftover=$(ls "$work_dir"/fail_output.txt* 2>/dev/null | wc -l)
    echo "  Leftover temp files: $leftover"

    rm -rf "$work_dir"
}

# === Exercise 5: Multi-Level Debug Logging Framework ===
# Problem: LOG_LEVEL env variable, log_error/warn/info/debug functions,
# timestamps, and log_structured with key=value pairs.
exercise_5() {
    echo "=== Exercise 5: Multi-Level Debug Logging Framework ==="

    # --- Inline log_lib.sh ---
    # Log levels: 0=none, 1=error, 2=warn, 3=info, 4=debug
    local LOG_LEVEL="${LOG_LEVEL:-3}"

    _log() {
        local level_num="$1"
        local level_name="$2"
        shift 2
        local message="$*"

        if (( level_num <= LOG_LEVEL )); then
            local timestamp
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            echo "  [$timestamp] [$level_name] $message"
        fi
    }

    log_error() { _log 1 "ERROR" "$*"; }
    log_warn()  { _log 2 "WARN " "$*"; }
    log_info()  { _log 3 "INFO " "$*"; }
    log_debug() { _log 4 "DEBUG" "$*"; }

    log_structured() {
        local level_num="$1"
        local level_name="$2"
        shift 2

        if (( level_num <= LOG_LEVEL )); then
            local timestamp
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            echo -n "  [$timestamp] [$level_name]"
            while [ $# -gt 0 ]; do
                echo -n " $1"
                shift
            done
            echo ""
        fi
    }

    # Test at different log levels
    echo "--- LOG_LEVEL=3 (info) ---"
    LOG_LEVEL=3
    log_error "Database connection failed"
    log_warn  "High memory usage: 85%"
    log_info  "Server started on port 8080"
    log_debug "Loading config from /etc/app.conf"  # Should NOT appear

    echo ""
    echo "--- LOG_LEVEL=1 (error only) ---"
    LOG_LEVEL=1
    log_error "Critical failure"
    log_warn  "This should NOT appear"
    log_info  "This should NOT appear"
    log_debug "This should NOT appear"

    echo ""
    echo "--- LOG_LEVEL=4 (all messages) ---"
    LOG_LEVEL=4
    log_error "Error message"
    log_warn  "Warning message"
    log_info  "Info message"
    log_debug "Debug message"

    echo ""
    echo "--- Structured logging ---"
    LOG_LEVEL=3
    log_structured 3 "INFO " "event=startup" "version=1.0.0" "pid=$$"
    log_structured 3 "INFO " "event=request" "method=GET" "path=/api/users" "status=200"
    log_structured 1 "ERROR" "event=error" "error=connection_refused" "host=db.example.com"
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
