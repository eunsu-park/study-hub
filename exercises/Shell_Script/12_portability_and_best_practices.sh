#!/bin/bash
# Exercises for Lesson 12: Portability and Best Practices
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Audit a Script for Portability Issues ===
# Problem: Identify every portability issue in a script fragment
# and provide the portable fix for each.
exercise_1() {
    echo "=== Exercise 1: Audit a Script for Portability Issues ==="

    echo "--- Original script fragment ---"
    cat << 'SCRIPT'
  #!/bin/bash
  which python3 > /dev/null
  result=`python3 -c "print(2**10)"`
  echo "Result: $result"
  ls *.log | while read file; do
      wc -l $file
  done
  stat -c%s report.txt
  function cleanup { rm -f /tmp/myapp.$$; }
SCRIPT

    echo ""
    echo "--- Issues and Fixes ---"
    echo ""

    echo "  Issue 1: 'which python3'"
    echo "  Problem: 'which' is not POSIX; behavior varies across systems."
    echo "  Fix:     command -v python3 > /dev/null 2>&1"
    echo ""

    echo "  Issue 2: result=\`python3 -c ...\`"
    echo "  Problem: Backtick syntax is deprecated and harder to nest."
    echo "  Fix:     result=\$(python3 -c \"print(2**10)\")"
    echo ""

    echo "  Issue 3: ls *.log | while read file"
    echo "  Problem: Parsing ls output is fragile (spaces, special chars)."
    echo "           Also, pipe creates a subshell so variables don't persist."
    echo "  Fix:     for file in *.log; do [ -f \"\$file\" ] || continue; ..."
    echo ""

    echo "  Issue 4: wc -l \$file (unquoted variable)"
    echo "  Problem: Word splitting if filename contains spaces."
    echo "  Fix:     wc -l \"\$file\""
    echo ""

    echo "  Issue 5: stat -c%s report.txt"
    echo "  Problem: stat -c is GNU (Linux). macOS uses stat -f%z."
    echo "  Fix:     wc -c < report.txt   (POSIX portable)"
    echo "           Or: if on Linux, stat -c%s; on macOS, stat -f%z"
    echo ""

    echo "  Issue 6: function cleanup { ... }"
    echo "  Problem: 'function' keyword is a Bash extension, not POSIX."
    echo "  Fix:     cleanup() { rm -f /tmp/myapp.\$\$; }"
    echo ""

    echo "  Issue 7: /tmp/myapp.\$\$ (predictable temp file name)"
    echo "  Problem: PID-based temp names are predictable (race conditions)."
    echo "  Fix:     tmpfile=\$(mktemp) && trap 'rm -f \"\$tmpfile\"' EXIT"
    echo ""

    # Demonstrate the fixed version
    echo "--- Fixed script ---"
    cat << 'FIXED'
  #!/bin/sh
  command -v python3 > /dev/null 2>&1 || { echo "python3 not found"; exit 1; }
  result=$(python3 -c "print(2**10)")
  echo "Result: $result"
  for file in *.log; do
      [ -f "$file" ] || continue
      wc -l "$file"
  done
  wc -c < report.txt
  cleanup() { rm -f "$tmpfile"; }
  tmpfile=$(mktemp)
  trap cleanup EXIT
FIXED
}

# === Exercise 2: Write a POSIX-Compatible Script ===
# Problem: Rewrite a bash-specific script to POSIX sh.
exercise_2() {
    echo "=== Exercise 2: Write a POSIX-Compatible Script ==="

    echo "--- Original Bash-specific script ---"
    cat << 'ORIGINAL'
  #!/bin/bash
  declare -a files=()
  for f in *.conf; do
      [[ -f "$f" ]] && files+=("$f")
  done

  process() {
      local name="$1"
      echo "Processing: $name"
      [[ "$name" =~ ^[0-9]+_ ]] && echo "  (numbered file)"
  }

  for f in "${files[@]}"; do
      process "$f"
  done
ORIGINAL

    echo ""
    echo "--- POSIX rewrite ---"
    cat << 'POSIX_VER'
  #!/bin/sh
  # No arrays in POSIX — use a newline-separated string or positional params
  files=""
  for f in *.conf; do
      [ -f "$f" ] || continue
      files="$files $f"
  done

  process() {
      name="$1"
      echo "Processing: $name"
      # No =~ in POSIX — use case or expr
      case "$name" in
          [0-9]*_*) echo "  (numbered file)" ;;
      esac
  }

  for f in $files; do
      process "$f"
  done
POSIX_VER

    echo ""
    echo "--- Key differences ---"
    echo "  1. declare -a / arrays    -> space-separated string or positional params"
    echo "  2. [[ -f \"\$f\" ]]         -> [ -f \"\$f\" ]"
    echo "  3. files+=(\"value\")       -> files=\"\$files value\""
    echo "  4. [[ \$x =~ regex ]]     -> case \$x in pattern) ... ;; esac"
    echo "  5. \${files[@]}            -> \$files (unquoted for splitting)"
    echo ""

    # Demonstrate the POSIX version actually works
    echo "--- Running POSIX simulation ---"
    local work_dir="/tmp/posix_test_$$"
    mkdir -p "$work_dir"
    echo "a" > "$work_dir/01_server.conf"
    echo "b" > "$work_dir/app.conf"
    echo "c" > "$work_dir/99_debug.conf"

    (
        cd "$work_dir"
        # POSIX version
        files=""
        for f in *.conf; do
            [ -f "$f" ] || continue
            files="$files $f"
        done

        process() {
            local name="$1"
            echo "  Processing: $name"
            case "$name" in
                [0-9]*_*) echo "    (numbered file)" ;;
            esac
        }

        for f in $files; do
            process "$f"
        done
    )

    rm -rf "$work_dir"
}

# === Exercise 3: Integrate ShellCheck into a Workflow ===
# Problem: Create .shellcheckrc, write lint.sh, test with intentional warnings.
exercise_3() {
    echo "=== Exercise 3: Integrate ShellCheck into a Workflow ==="

    local work_dir="/tmp/shellcheck_ex_$$"
    mkdir -p "$work_dir"

    # Step 1: Create .shellcheckrc
    cat > "$work_dir/.shellcheckrc" << 'EOF'
shell=bash
disable=SC2034
EOF
    echo "--- .shellcheckrc ---"
    cat "$work_dir/.shellcheckrc" | sed 's/^/  /'

    # Step 2: Create lint.sh
    cat > "$work_dir/lint.sh" << 'LINT'
#!/bin/bash
# lint.sh - Find all .sh files and run shellcheck on each.
# Exits with code 1 if any errors (severity=error) are found.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
error_count=0

# Find all .sh files
while IFS= read -r -d '' script; do
    echo "Checking: $script"
    if ! shellcheck --severity=error "$script"; then
        ((error_count++))
    fi
done < <(find "$SCRIPT_DIR" -name "*.sh" -not -name "lint.sh" -print0)

if (( error_count > 0 )); then
    echo ""
    echo "FAIL: $error_count file(s) have errors."
    exit 1
else
    echo ""
    echo "PASS: All scripts clean."
    exit 0
fi
LINT
    chmod +x "$work_dir/lint.sh"

    echo ""
    echo "--- lint.sh ---"
    cat "$work_dir/lint.sh" | sed 's/^/  /'

    # Step 3: Create a test script with intentional warnings
    cat > "$work_dir/buggy.sh" << 'BUGGY'
#!/bin/bash
# This script has intentional ShellCheck warnings

# SC2086 (error): Double quote to prevent globbing and word splitting
filename="my file.txt"
cat $filename

# SC2046 (warning): Quote to prevent word splitting
files=$(ls *.txt)
echo $files
BUGGY

    echo ""
    echo "--- buggy.sh (intentional warnings) ---"
    cat "$work_dir/buggy.sh" | sed 's/^/  /'

    # Step 4: Run shellcheck (if available)
    echo ""
    echo "--- Running ShellCheck ---"
    if command -v shellcheck &>/dev/null; then
        echo "  ShellCheck found: $(shellcheck --version 2>/dev/null | head -2 | tail -1)"
        echo ""
        echo "  Checking buggy.sh:"
        shellcheck "$work_dir/buggy.sh" 2>&1 | sed 's/^/    /' || true
    else
        echo "  ShellCheck not installed. To install:"
        echo "    macOS:  brew install shellcheck"
        echo "    Linux:  apt-get install shellcheck"
        echo ""
        echo "  Simulating expected output:"
        echo "    In buggy.sh line 6:"
        echo "    cat \$filename"
        echo "        ^--------^ SC2086: Double quote to prevent globbing"
    fi

    rm -rf "$work_dir"
}

# === Exercise 4: Apply Performance Best Practices ===
# Problem: Profile and optimize a slow script using at least 3 optimizations.
exercise_4() {
    echo "=== Exercise 4: Apply Performance Best Practices ==="

    local work_dir="/tmp/perfopt_$$"
    mkdir -p "$work_dir"

    # Create a test log file
    local log_file="$work_dir/application.log"
    for i in $(seq 1 500); do
        echo "2024-01-15 10:00:$(printf '%02d' $((i % 60))) INFO Request processed"
        if (( i % 7 == 0 )); then
            echo "2024-01-15 10:00:$(printf '%02d' $((i % 60))) ERROR Connection timeout"
        fi
        if (( i % 13 == 0 )); then
            echo "2024-01-15 10:00:$(printf '%02d' $((i % 60))) ERROR Disk full"
        fi
    done > "$log_file"

    echo "  Log file: $(wc -l < "$log_file") lines"
    echo ""

    # --- Original (slow) version ---
    echo "--- Version 1: Original (slow) ---"
    local start_time=$SECONDS
    count=0
    while read -r line; do
        if echo "$line" | grep -q "ERROR"; then
            count=$((count + 1))
        fi
    done < "$log_file"
    local v1_time=$(( SECONDS - start_time ))
    echo "  Error count: $count"
    echo "  Time: ${v1_time}s"

    # --- Optimization 1: Use grep -c instead of loop ---
    echo ""
    echo "--- Version 2: grep -c (avoid loop entirely) ---"
    start_time=$SECONDS
    count=$(grep -c "ERROR" "$log_file")
    local v2_time=$(( SECONDS - start_time ))
    echo "  Error count: $count"
    echo "  Time: ${v2_time}s"
    echo "  Optimization: Replace while-loop+grep with single 'grep -c'"

    # --- Optimization 2: Built-in string matching instead of external grep ---
    echo ""
    echo "--- Version 3: Built-in [[ == *pattern* ]] ---"
    start_time=$SECONDS
    count=0
    while IFS= read -r line; do
        if [[ "$line" == *ERROR* ]]; then
            (( count++ ))
        fi
    done < "$log_file"
    local v3_time=$(( SECONDS - start_time ))
    echo "  Error count: $count"
    echo "  Time: ${v3_time}s"
    echo "  Optimization: Use bash built-in pattern matching in loop"

    # --- Optimization 3: awk for counting ---
    echo ""
    echo "--- Version 4: awk single-pass ---"
    start_time=$SECONDS
    count=$(awk '/ERROR/ {c++} END {print c}' "$log_file")
    local v4_time=$(( SECONDS - start_time ))
    echo "  Error count: $count"
    echo "  Time: ${v4_time}s"
    echo "  Optimization: Single awk invocation does everything"

    echo ""
    echo "--- Summary of Optimizations ---"
    echo "  1. grep -c: Replace read loop + external grep with single grep -c"
    echo "  2. Built-in: Use [[ \$line == *pattern* ]] instead of piping to grep"
    echo "  3. awk:      Single-pass counting with awk, no shell loop overhead"
    echo ""
    echo "  Best practice: For simple counting, 'grep -c' or 'awk' is fastest."
    echo "  Avoid spawning external commands inside loops."

    rm -rf "$work_dir"
}

# === Exercise 5: Create a Distributable Script Package ===
# Problem: install.sh with --prefix, uninstall.sh, version check.
exercise_5() {
    echo "=== Exercise 5: Create a Distributable Script Package ==="

    local work_dir="/tmp/distpkg_$$"
    mkdir -p "$work_dir"

    # Create the main script
    cat > "$work_dir/myscript.sh" << 'APP'
#!/bin/bash
# myscript.sh - A sample distributable script
VERSION="1.2.0"

main() {
    case "${1:-}" in
        --version|-V) echo "myscript version $VERSION" ;;
        --help|-h)    echo "Usage: myscript [OPTIONS]"; echo "  -V  Show version"; echo "  -h  Show help" ;;
        *)            echo "Hello from myscript $VERSION!" ;;
    esac
}
main "$@"
APP
    chmod +x "$work_dir/myscript.sh"

    # Create install.sh
    cat > "$work_dir/install.sh" << 'INSTALLER'
#!/bin/bash
set -euo pipefail

SCRIPT_NAME="myscript"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="/usr/local"

# Parse --prefix
while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --prefix=*) PREFIX="${1#*=}"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

BIN_DIR="$PREFIX/bin"
MAN_DIR="$PREFIX/share/man/man1"

# Bash version check
BASH_MAJOR="${BASH_VERSINFO[0]}"
if (( BASH_MAJOR < 4 )); then
    echo "WARNING: Bash $BASH_VERSION detected. This script requires Bash 4.0+."
    echo "Please upgrade bash before using $SCRIPT_NAME."
    exit 1
fi

echo "Installing $SCRIPT_NAME to $PREFIX..."

# Create directories
mkdir -p "$BIN_DIR"
mkdir -p "$MAN_DIR"

# Copy script
cp "$SCRIPT_DIR/$SCRIPT_NAME.sh" "$BIN_DIR/$SCRIPT_NAME"
chmod 755 "$BIN_DIR/$SCRIPT_NAME"
echo "  Installed: $BIN_DIR/$SCRIPT_NAME"

# Generate man page
cat > "$MAN_DIR/$SCRIPT_NAME.1" << MANPAGE
.TH MYSCRIPT 1 "$(date +'%B %Y')" "myscript 1.2.0" "User Commands"
.SH NAME
myscript \- A sample distributable script
.SH SYNOPSIS
.B myscript
[\fIOPTIONS\fR]
.SH OPTIONS
.TP
.BR \-V ", " \-\-version
Show version information
.TP
.BR \-h ", " \-\-help
Show help message
.SH EXIT STATUS
.TP
.B 0
Success
MANPAGE
echo "  Installed: $MAN_DIR/$SCRIPT_NAME.1"

echo "Installation complete!"
INSTALLER
    chmod +x "$work_dir/install.sh"

    # Create uninstall.sh
    cat > "$work_dir/uninstall.sh" << 'UNINSTALLER'
#!/bin/bash
set -euo pipefail

SCRIPT_NAME="myscript"
PREFIX="/usr/local"

# Parse --prefix
while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --prefix=*) PREFIX="${1#*=}"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

BIN_DIR="$PREFIX/bin"
MAN_DIR="$PREFIX/share/man/man1"

echo "Uninstalling $SCRIPT_NAME from $PREFIX..."

# Remove script
if [ -f "$BIN_DIR/$SCRIPT_NAME" ]; then
    rm -f "$BIN_DIR/$SCRIPT_NAME"
    echo "  Removed: $BIN_DIR/$SCRIPT_NAME"
else
    echo "  Not found: $BIN_DIR/$SCRIPT_NAME"
fi

# Remove man page
if [ -f "$MAN_DIR/$SCRIPT_NAME.1" ]; then
    rm -f "$MAN_DIR/$SCRIPT_NAME.1"
    echo "  Removed: $MAN_DIR/$SCRIPT_NAME.1"
else
    echo "  Not found: $MAN_DIR/$SCRIPT_NAME.1"
fi

echo "Uninstallation complete!"
UNINSTALLER
    chmod +x "$work_dir/uninstall.sh"

    # Test with --prefix pointing to temp location
    local install_prefix="$work_dir/installed"

    echo "--- install.sh content (excerpt) ---"
    head -20 "$work_dir/install.sh" | sed 's/^/  /'
    echo "  ..."

    echo ""
    echo "--- Running install.sh --prefix $install_prefix ---"
    bash "$work_dir/install.sh" --prefix "$install_prefix"

    echo ""
    echo "--- Verify installation ---"
    echo "  Files:"
    ls -la "$install_prefix/bin/myscript" 2>/dev/null | sed 's/^/    /'
    ls -la "$install_prefix/share/man/man1/myscript.1" 2>/dev/null | sed 's/^/    /'

    echo ""
    echo "  Running installed script:"
    "$install_prefix/bin/myscript" --version | sed 's/^/    /'
    "$install_prefix/bin/myscript" | sed 's/^/    /'

    echo ""
    echo "--- Running uninstall.sh --prefix $install_prefix ---"
    bash "$work_dir/uninstall.sh" --prefix "$install_prefix"

    echo ""
    echo "--- Verify uninstallation ---"
    if [ ! -f "$install_prefix/bin/myscript" ]; then
        echo "  Confirmed: myscript removed"
    fi
    if [ ! -f "$install_prefix/share/man/man1/myscript.1" ]; then
        echo "  Confirmed: man page removed"
    fi

    rm -rf "$work_dir"
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
