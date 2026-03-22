#!/usr/bin/env bash

# Logging Library
# Provides colored, leveled logging with optional file output

# Color codes
declare -r COLOR_RESET='\033[0m'
declare -r COLOR_RED='\033[0;31m'
declare -r COLOR_YELLOW='\033[0;33m'
declare -r COLOR_GREEN='\033[0;32m'
declare -r COLOR_BLUE='\033[0;34m'
declare -r COLOR_GRAY='\033[0;90m'

# Log levels
declare -r LEVEL_DEBUG=0
declare -r LEVEL_INFO=1
declare -r LEVEL_WARN=2
declare -r LEVEL_ERROR=3

# Why: : "${VAR:=default}" is the idiomatic way to set a default only if unset.
# The : (no-op) consumes the expansion, and := assigns if the variable is empty.
: "${LOG_LEVEL:=$LEVEL_INFO}"

# Optional log file (empty = no file logging)
: "${LOG_FILE:=}"

# Get current timestamp
_log_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Internal logging function
_log() {
    local level="$1"
    local level_num="$2"
    local color="$3"
    shift 3
    local message="$*"

    # Check if this message should be logged based on LOG_LEVEL
    if [[ $level_num -lt $LOG_LEVEL ]]; then
        return 0
    fi

    local timestamp
    timestamp=$(_log_timestamp)

    # Format: [TIMESTAMP] LEVEL: message
    local formatted_console="${color}[${timestamp}] ${level}:${COLOR_RESET} ${message}"
    local formatted_file="[${timestamp}] ${level}: ${message}"

    # Why: log output goes to stderr so it never mixes with a function's
    # stdout return value — callers can capture data while still seeing logs.
    echo -e "$formatted_console" >&2

    # Output to file if LOG_FILE is set
    if [[ -n "$LOG_FILE" ]]; then
        echo "$formatted_file" >> "$LOG_FILE"
    fi
}

# Public logging functions

log_debug() {
    _log "DEBUG" "$LEVEL_DEBUG" "$COLOR_GRAY" "$@"
}

log_info() {
    _log "INFO " "$LEVEL_INFO" "$COLOR_GREEN" "$@"
}

log_warn() {
    _log "WARN " "$LEVEL_WARN" "$COLOR_YELLOW" "$@"
}

log_error() {
    _log "ERROR" "$LEVEL_ERROR" "$COLOR_RED" "$@"
}

# Set log level by name
set_log_level() {
    local level_name="$1"

    case "${level_name^^}" in
        DEBUG)
            LOG_LEVEL=$LEVEL_DEBUG
            ;;
        INFO)
            LOG_LEVEL=$LEVEL_INFO
            ;;
        WARN|WARNING)
            LOG_LEVEL=$LEVEL_WARN
            ;;
        ERROR)
            LOG_LEVEL=$LEVEL_ERROR
            ;;
        *)
            log_error "Invalid log level: $level_name (use DEBUG, INFO, WARN, ERROR)"
            return 1
            ;;
    esac

    log_info "Log level set to: $level_name"
}

# Enable file logging
enable_file_logging() {
    local file="$1"

    # Create log file or append if exists
    if touch "$file" 2>/dev/null; then
        LOG_FILE="$file"
        log_info "File logging enabled: $file"
    else
        log_error "Cannot write to log file: $file"
        return 1
    fi
}

# Disable file logging
disable_file_logging() {
    if [[ -n "$LOG_FILE" ]]; then
        log_info "File logging disabled"
        LOG_FILE=""
    fi
}

# Print a separator line for visual organization
log_separator() {
    local char="${1:--}"
    local length="${2:-60}"

    # Why: printf -v writes to a variable without a subshell, then global
    # substitution replaces every space with the chosen char — pure bash, no seq.
    local line
    printf -v line '%*s' "$length"
    line="${line// /$char}"

    echo -e "${COLOR_BLUE}${line}${COLOR_RESET}" >&2

    if [[ -n "$LOG_FILE" ]]; then
        echo "$line" >> "$LOG_FILE"
    fi
}

# Log a section header
log_section() {
    local title="$1"

    log_separator "="
    log_info "$title"
    log_separator "="
}
