#!/usr/bin/env bash

# Validation Library
# Provides functions for validating common data types and formats

# Validate email address
validate_email() {
    local email="$1"

    # Basic email regex pattern
    local email_regex='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if [[ "$email" =~ $email_regex ]]; then
        return 0
    else
        return 1
    fi
}

# Validate IPv4 address
validate_ip() {
    local ip="$1"

    # IPv4 regex pattern
    local ip_regex='^([0-9]{1,3}\.){3}[0-9]{1,3}$'

    if [[ ! "$ip" =~ $ip_regex ]]; then
        return 1
    fi

    # Why: setting IFS locally to '.' splits the IP string into octets via
    # unquoted expansion — scoped to this function so global IFS is untouched.
    local IFS='.'
    local -a octets=($ip)

    for octet in "${octets[@]}"; do
        if [[ $octet -gt 255 ]]; then
            return 1
        fi
    done

    return 0
}

# Validate port number
validate_port() {
    local port="$1"

    if ! is_integer "$port"; then
        return 1
    fi

    if [[ $port -lt 1 || $port -gt 65535 ]]; then
        return 1
    fi

    return 0
}

# Validate file/directory path exists
validate_path() {
    local path="$1"
    local type="${2:-any}"  # any, file, dir

    case "$type" in
        file)
            [[ -f "$path" ]]
            ;;
        dir)
            [[ -d "$path" ]]
            ;;
        any)
            [[ -e "$path" ]]
            ;;
        *)
            echo "Error: Invalid path type: $type (use 'file', 'dir', or 'any')" >&2
            return 2
            ;;
    esac
}

# Check if value is an integer
is_integer() {
    local value="$1"

    [[ "$value" =~ ^-?[0-9]+$ ]]
}

# Check if value is a positive integer
is_positive() {
    local value="$1"

    if ! is_integer "$value"; then
        return 1
    fi

    [[ $value -gt 0 ]]
}

# Check if value is non-negative (>= 0)
is_non_negative() {
    local value="$1"

    if ! is_integer "$value"; then
        return 1
    fi

    [[ $value -ge 0 ]]
}

# Check if value is in range [min, max]
is_in_range() {
    local value="$1"
    local min="$2"
    local max="$3"

    if ! is_integer "$value"; then
        return 1
    fi

    [[ $value -ge $min && $value -le $max ]]
}

# Validate URL
validate_url() {
    local url="$1"

    # Basic URL regex pattern
    local url_regex='^https?://[a-zA-Z0-9.-]+(:[0-9]+)?(/.*)?$'

    [[ "$url" =~ $url_regex ]]
}

# Validate hostname
validate_hostname() {
    local hostname="$1"

    # Hostname regex: letters, numbers, hyphens, dots
    # Cannot start/end with hyphen, max 253 chars
    local hostname_regex='^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'

    if [[ ${#hostname} -gt 253 ]]; then
        return 1
    fi

    [[ "$hostname" =~ $hostname_regex ]]
}

# Validate MAC address
validate_mac() {
    local mac="$1"

    # MAC address regex (colon or hyphen separated)
    local mac_regex='^([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$'

    [[ "$mac" =~ $mac_regex ]]
}

# Check if string is empty or only whitespace
is_blank() {
    local value="$1"

    [[ -z "${value// /}" ]]
}

# Check if string is not empty
is_not_blank() {
    local value="$1"

    ! is_blank "$value"
}

# Validate date in YYYY-MM-DD format
validate_date() {
    local date_str="$1"

    # Check format
    local date_regex='^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
    if [[ ! "$date_str" =~ $date_regex ]]; then
        return 1
    fi

    # Why: GNU date uses -d while macOS/BSD uses -j -f. Trying both makes
    # this validation portable across Linux and macOS without platform checks.
    if date -d "$date_str" >/dev/null 2>&1; then
        return 0
    elif date -j -f "%Y-%m-%d" "$date_str" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Validate hex color code
validate_hex_color() {
    local color="$1"

    # Hex color regex (#RGB or #RRGGBB)
    local hex_regex='^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'

    [[ "$color" =~ $hex_regex ]]
}

# Check if command exists in PATH
command_exists() {
    local cmd="$1"

    # Why: command -v is POSIX-portable and avoids the hash/type pitfalls.
    # It checks the actual PATH resolution, not shell builtins or aliases.
    command -v "$cmd" >/dev/null 2>&1
}

# Validate semantic version (e.g., 1.2.3)
validate_semver() {
    local version="$1"

    local semver_regex='^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$'

    [[ "$version" =~ $semver_regex ]]
}

# Require value to be non-empty (for required arguments)
require_value() {
    local value="$1"
    local field_name="${2:-value}"

    if is_blank "$value"; then
        echo "Error: $field_name is required" >&2
        return 1
    fi

    return 0
}
