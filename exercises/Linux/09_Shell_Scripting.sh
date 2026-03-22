#!/bin/bash
# Exercises for Lesson 09: Shell Scripting
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: Variables, Quoting, and Parameter Expansion ===
# Problem: Demonstrate proper variable usage, quoting rules, and parameter expansion
#          techniques for robust shell scripts.
exercise_1() {
    echo "=== Exercise 1: Variables, Quoting, and Parameter Expansion ==="
    echo ""
    echo "Scenario: You are writing a deployment script that must handle file paths"
    echo "with spaces, provide default values for unconfigured variables, and safely"
    echo "manipulate strings."
    echo ""

    echo "--- Part A: Variable Assignment and Quoting Rules ---"
    echo "Solution:"
    echo "  APP_NAME=\"my-webapp\"                # No spaces around '='"
    echo "  DEPLOY_DIR=\"/opt/apps/\${APP_NAME}\"   # Double quotes preserve variable expansion"
    echo "  GREETING='Hello \$USER'               # Single quotes: literal string, no expansion"
    echo "  COMMAND=\"Files in \$(ls /tmp | wc -l)\" # \$() for command substitution inside double quotes"
    echo ""
    echo "  Explanation:"
    echo "    - Double quotes (\"\") expand variables and command substitutions"
    echo "    - Single quotes ('') treat everything as literal text"
    echo "    - Always quote variables in commands: rm \"\$file\" not rm \$file"
    echo "    - Unquoted variables undergo word splitting and glob expansion"
    echo ""

    echo "--- Part B: Parameter Expansion with Defaults ---"
    echo "Solution:"
    echo "  LOG_DIR=\${LOG_DIR:-/var/log/myapp}     # Use default if LOG_DIR is unset or empty"
    echo "  DB_HOST=\${DB_HOST:=localhost}           # Assign default if unset or empty"
    echo "  PORT=\${PORT:?\"PORT must be set\"}        # Exit with error if unset or empty"
    echo "  OPTIONAL=\${OPTIONAL:+\"--flag \$OPTIONAL\"} # Use alternate value only if set and non-empty"
    echo ""
    echo "  Explanation:"
    echo "    :- provides a default without modifying the variable"
    echo "    := provides a default AND assigns it to the variable"
    echo "    :? aborts with an error message if the variable is unset"
    echo "    :+ returns the alternate value only when the variable IS set"
    echo ""

    echo "--- Part C: String Manipulation with Parameter Expansion ---"
    echo "Solution:"
    echo "  FILE=\"/home/user/documents/report.tar.gz\""
    echo "  echo \${FILE##*/}       # report.tar.gz  (remove longest prefix matching */)"
    echo "  echo \${FILE%.*}        # /home/user/documents/report.tar  (remove shortest suffix)"
    echo "  echo \${FILE%%.*}       # /home/user/documents/report  (remove longest suffix)"
    echo "  echo \${FILE#*/}        # home/user/documents/report.tar.gz  (remove shortest prefix)"
    echo "  echo \${#FILE}          # String length"
    echo "  echo \${FILE:6:4}       # Substring: 'user' (offset:length)"
    echo "  echo \${FILE/report/summary}  # Replace first match"
    echo ""
    echo "  Explanation:"
    echo "    # and ## strip from the left (prefix); % and %% strip from the right (suffix)"
    echo "    Single symbol = shortest match; double symbol = longest match"
    echo ""

    echo "--- Verification ---"
    echo "  # Test quoting behavior:"
    echo "  set -x           # Enable debug mode to see expansions"
    echo "  echo \"\$HOME\"     # Shows expanded value"
    echo "  echo '\$HOME'     # Shows literal \$HOME"
    echo "  set +x           # Disable debug mode"
}

# === Exercise 2: Conditionals and Loops ===
# Problem: Implement conditional logic with test operators and loop constructs
#          for batch processing tasks.
exercise_2() {
    echo "=== Exercise 2: Conditionals and Loops ==="
    echo ""
    echo "Scenario: Write a log rotation script that checks file sizes, processes"
    echo "log files in a directory, and handles different log types."
    echo ""

    echo "--- Part A: Conditionals with [[ ]] and File Tests ---"
    echo "Solution:"
    cat << 'SCRIPT'
  # Check if log directory exists and is writable
  if [[ -d "/var/log/myapp" && -w "/var/log/myapp" ]]; then
      echo "Log directory is ready"
  elif [[ -d "/var/log/myapp" ]]; then
      echo "Log directory exists but is not writable"
      exit 1
  else
      mkdir -p /var/log/myapp
  fi

  # Check file size (rotate if > 10MB)
  FILE_SIZE=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
  if (( FILE_SIZE > 10485760 )); then
      echo "Rotating: $LOG_FILE ($FILE_SIZE bytes)"
  fi
SCRIPT
    echo ""
    echo "  Explanation:"
    echo "    [[ ]] is preferred over [ ] for conditionals (supports && ||, no word splitting)"
    echo "    -d tests directory existence; -w tests write permission; -f tests regular file"
    echo "    (( )) is used for arithmetic comparisons (>, <, ==, >=)"
    echo ""

    echo "--- Part B: For Loops and While Loops ---"
    echo "Solution:"
    cat << 'SCRIPT'
  # C-style for loop with counter
  for (( i=1; i<=5; i++ )); do
      ARCHIVE="app.log.${i}.gz"
      echo "Checking archive: $ARCHIVE"
  done

  # For loop over glob pattern
  for logfile in /var/log/myapp/*.log; do
      [[ -f "$logfile" ]] || continue    # Skip if glob didn't match
      echo "Processing: $logfile"
  done

  # While loop reading lines from a file
  while IFS= read -r line; do
      echo "Config entry: $line"
  done < /etc/myapp/rotation.conf
SCRIPT
    echo ""
    echo "  Explanation:"
    echo "    IFS= prevents leading/trailing whitespace trimming"
    echo "    -r prevents backslash interpretation in read"
    echo "    'continue' skips to the next iteration; 'break' exits the loop"
    echo ""

    echo "--- Part C: Case Statements for Pattern Matching ---"
    echo "Solution:"
    cat << 'SCRIPT'
  case "$LOG_TYPE" in
      access|request)
          RETENTION=30
          COMPRESS=true
          ;;
      error|warn*)
          RETENTION=90
          COMPRESS=true
          ;;
      debug)
          RETENTION=7
          COMPRESS=false
          ;;
      *)
          echo "Unknown log type: $LOG_TYPE" >&2
          exit 1
          ;;
  esac
  echo "Retention: ${RETENTION} days, Compress: ${COMPRESS}"
SCRIPT
    echo ""
    echo "  Explanation:"
    echo "    case supports glob patterns (*, ?, []) and alternation with |"
    echo "    ;; terminates each branch; ;& falls through to next branch"
    echo "    *) is the default/catch-all branch"
    echo ""

    echo "--- Verification ---"
    echo "  bash -n script.sh    # Syntax check without executing"
    echo "  shellcheck script.sh # Static analysis for common pitfalls"
}

# === Exercise 3: Functions, Error Handling, and Script Structure ===
# Problem: Build a well-structured script with functions, proper error handling,
#          and the set -euo pipefail safety net.
exercise_3() {
    echo "=== Exercise 3: Functions, Error Handling, and Script Structure ==="
    echo ""
    echo "Scenario: Create a production-quality deployment script with proper"
    echo "error handling, logging, cleanup on exit, and modular functions."
    echo ""

    echo "--- Part A: Strict Mode and Error Handling ---"
    echo "Solution:"
    cat << 'SCRIPT'
  #!/bin/bash
  set -euo pipefail    # The unofficial bash strict mode

  # set -e:          Exit immediately on any command failure
  # set -u:          Treat unset variables as errors
  # set -o pipefail: Pipe fails if ANY command in the pipeline fails
  #                  (default: only last command's exit code matters)

  # Trap for cleanup on exit (normal or error)
  cleanup() {
      local exit_code=$?
      echo "Cleaning up temporary files..."
      rm -f "$TEMP_FILE" 2>/dev/null
      exit "$exit_code"    # Preserve original exit code
  }
  trap cleanup EXIT       # Runs on EXIT (covers normal exit + errors + signals)
  trap 'echo "Interrupted!"; exit 130' INT TERM
SCRIPT
    echo ""
    echo "  Explanation:"
    echo "    trap ... EXIT runs cleanup regardless of how the script ends"
    echo "    trap ... INT TERM handles Ctrl+C (SIGINT) and kill (SIGTERM)"
    echo "    Capturing \$? at the start of cleanup preserves the original exit code"
    echo ""

    echo "--- Part B: Functions with Local Variables and Return Values ---"
    echo "Solution:"
    cat << 'SCRIPT'
  log() {
      local level="${1:?'log level required'}"
      local message="${2:?'log message required'}"
      local timestamp
      timestamp=$(date '+%Y-%m-%d %H:%M:%S')
      printf '[%s] [%-5s] %s\n' "$timestamp" "$level" "$message"
  }

  validate_environment() {
      local missing=0
      for cmd in git docker curl; do
          if ! command -v "$cmd" &>/dev/null; then
              log "ERROR" "Required command not found: $cmd"
              ((missing++))
          fi
      done
      return "$missing"    # 0 = success, non-zero = number of missing commands
  }

  # Usage with error checking:
  if ! validate_environment; then
      log "FATAL" "Missing dependencies, aborting"
      exit 1
  fi
SCRIPT
    echo ""
    echo "  Explanation:"
    echo "    'local' scopes variables to the function (prevents pollution)"
    echo "    'command -v' checks if a command exists (portable, preferred over 'which')"
    echo "    Functions return exit codes (0-255); use stdout for string return values"
    echo ""

    echo "--- Part C: Complete Script Template ---"
    echo "Solution:"
    cat << 'SCRIPT'
  #!/bin/bash
  set -euo pipefail

  # --- Constants ---
  readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

  # --- Defaults (overridable by environment) ---
  VERBOSE="${VERBOSE:-false}"
  DRY_RUN="${DRY_RUN:-false}"

  # --- Functions ---
  usage() {
      cat <<EOF
  Usage: $SCRIPT_NAME [OPTIONS] <target>
  Options:
    -v, --verbose    Enable verbose output
    -n, --dry-run    Show what would be done
    -h, --help       Show this help
  EOF
  }

  # --- Argument Parsing ---
  while [[ $# -gt 0 ]]; do
      case "$1" in
          -v|--verbose) VERBOSE=true; shift ;;
          -n|--dry-run) DRY_RUN=true; shift ;;
          -h|--help) usage; exit 0 ;;
          --) shift; break ;;
          -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
          *) break ;;
      esac
  done

  TARGET="${1:?'target argument required (use --help for usage)'}"

  # --- Main Logic ---
  main() {
      log "INFO" "Starting deployment of $TARGET"
      # ... deployment steps here ...
      log "INFO" "Deployment complete"
  }

  main "$@"
SCRIPT
    echo ""
    echo "  Explanation:"
    echo "    BASH_SOURCE[0] reliably gives the script path (works with source and symlinks)"
    echo "    'readonly' prevents accidental reassignment of constants"
    echo "    Argument parsing loop with shift handles both short and long options"
    echo ""

    echo "--- Verification ---"
    echo "  bash -x script.sh      # Trace mode: shows each command before execution"
    echo "  bash -n script.sh      # Syntax-only check"
    echo "  shellcheck script.sh   # Lint for common bugs and portability issues"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
