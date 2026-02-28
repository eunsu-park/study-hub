#!/bin/bash
# Exercises for Lesson 08: Regular Expressions in Bash
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Advanced Input Validator ===
# Problem: Validation library with multiple rules per field,
# custom validators, detailed error messages, and conditional validation.
exercise_1() {
    echo "=== Exercise 1: Advanced Input Validator ==="

    declare -a validation_errors=()

    # Core rule validators
    rule_required() {
        local value="$1"
        [ -n "$value" ]
    }

    rule_type_email() {
        local value="$1"
        local pattern='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        [[ "$value" =~ $pattern ]]
    }

    rule_type_int() {
        local value="$1"
        [[ "$value" =~ ^-?[0-9]+$ ]]
    }

    rule_min_length() {
        local value="$1" min="$2"
        (( ${#value} >= min ))
    }

    rule_max_length() {
        local value="$1" max="$2"
        (( ${#value} <= max ))
    }

    rule_pattern() {
        local value="$1" pat="$2"
        [[ "$value" =~ $pat ]]
    }

    rule_range() {
        local value="$1" min="$2" max="$3"
        [[ "$value" =~ ^-?[0-9]+$ ]] && (( value >= min && value <= max ))
    }

    # Password strength validator
    rule_password_strength() {
        local value="$1"
        local issues=()
        (( ${#value} < 8 )) && issues+=("at least 8 characters")
        [[ ! "$value" =~ [A-Z] ]] && issues+=("an uppercase letter")
        [[ ! "$value" =~ [a-z] ]] && issues+=("a lowercase letter")
        [[ ! "$value" =~ [0-9] ]] && issues+=("a digit")
        [[ ! "$value" =~ [^a-zA-Z0-9] ]] && issues+=("a special character")
        if (( ${#issues[@]} > 0 )); then
            local IFS=', '
            echo "needs ${issues[*]}"
            return 1
        fi
        return 0
    }

    # Validate a single field against multiple rules
    validate_field() {
        local field_name="$1"
        local value="$2"
        shift 2
        local errors=0

        while [ $# -gt 0 ]; do
            local rule="$1"; shift
            local msg=""

            case "$rule" in
                required)
                    if ! rule_required "$value"; then
                        validation_errors+=("  $field_name: required")
                        (( errors++ ))
                    fi
                    ;;
                email)
                    if [ -n "$value" ] && ! rule_type_email "$value"; then
                        validation_errors+=("  $field_name: invalid email format")
                        (( errors++ ))
                    fi
                    ;;
                int)
                    if [ -n "$value" ] && ! rule_type_int "$value"; then
                        validation_errors+=("  $field_name: must be an integer")
                        (( errors++ ))
                    fi
                    ;;
                min:*)
                    local min="${rule#min:}"
                    if [ -n "$value" ] && ! rule_min_length "$value" "$min"; then
                        validation_errors+=("  $field_name: min length is $min")
                        (( errors++ ))
                    fi
                    ;;
                max:*)
                    local max="${rule#max:}"
                    if [ -n "$value" ] && ! rule_max_length "$value" "$max"; then
                        validation_errors+=("  $field_name: max length is $max")
                        (( errors++ ))
                    fi
                    ;;
                range:*-*)
                    local spec="${rule#range:}"
                    local rmin="${spec%-*}" rmax="${spec#*-}"
                    if [ -n "$value" ] && ! rule_range "$value" "$rmin" "$rmax"; then
                        validation_errors+=("  $field_name: must be in range $rmin-$rmax")
                        (( errors++ ))
                    fi
                    ;;
                password)
                    if [ -n "$value" ]; then
                        msg=$(rule_password_strength "$value") || {
                            validation_errors+=("  $field_name: $msg")
                            (( errors++ ))
                        }
                    fi
                    ;;
            esac
        done

        return $errors
    }

    # Print validation report
    print_report() {
        if (( ${#validation_errors[@]} == 0 )); then
            echo "  [PASS] All validations passed"
        else
            echo "  [FAIL] ${#validation_errors[@]} error(s):"
            for err in "${validation_errors[@]}"; do
                echo "    - $err"
            done
        fi
    }

    # Test: valid record
    echo "--- Record 1 (valid) ---"
    validation_errors=()
    validate_field "name"     "Alice"             required min:2 max:50
    validate_field "email"    "alice@example.com"  required email
    validate_field "age"      "30"                 required int range:0-150
    validate_field "password" "MyP@ssw0rd!"        required password
    print_report

    echo ""
    echo "--- Record 2 (invalid) ---"
    validation_errors=()
    validate_field "name"     ""                   required
    validate_field "email"    "not-an-email"       required email
    validate_field "age"      "abc"                required int
    validate_field "password" "weak"               required password
    print_report
}

# === Exercise 2: Log Parser with Regex ===
# Problem: Auto-detect log format, extract fields, handle multi-line entries,
# validate format, convert timestamps, and generate statistics.
exercise_2() {
    echo "=== Exercise 2: Log Parser with Regex ==="

    local log_file="/tmp/logregex_$$.log"
    cat > "$log_file" << 'EOF'
192.168.1.10 - - [13/Feb/2024:10:00:01 +0000] "GET /index.html HTTP/1.1" 200 1234
192.168.1.20 - - [13/Feb/2024:10:01:15 +0000] "POST /api/login HTTP/1.1" 401 98
192.168.1.10 - - [13/Feb/2024:10:02:30 +0000] "GET /api/users HTTP/1.1" 500 512
192.168.1.30 - - [13/Feb/2024:10:03:00 +0000] "GET /favicon.ico HTTP/1.1" 404 0
192.168.1.20 - - [13/Feb/2024:10:05:22 +0000] "POST /api/login HTTP/1.1" 200 256
192.168.1.10 - - [13/Feb/2024:10:06:00 +0000] "GET /api/data HTTP/1.1" 200 4096
2024-02-13 10:10:00 [ERROR] Database connection failed
2024-02-13 10:11:00 [INFO] Reconnecting to database
2024-02-13 10:12:00 [WARN] High latency detected: 1500ms
EOF

    # Detect log format based on patterns
    detect_format() {
        local line="$1"
        # Apache/Nginx combined log format
        local apache_pat='^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ .+ \[.+\] ".+" [0-9]+ [0-9]+'
        # Syslog-like format
        local syslog_pat='^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} \[(INFO|WARN|ERROR|DEBUG)\]'

        if [[ "$line" =~ $apache_pat ]]; then
            echo "apache"
        elif [[ "$line" =~ $syslog_pat ]]; then
            echo "syslog"
        else
            echo "unknown"
        fi
    }

    # Parse Apache log line
    parse_apache() {
        local line="$1"
        local pat='^([0-9.]+) [^ ]+ [^ ]+ \[([^\]]+)\] "([A-Z]+) ([^ ]+) [^"]*" ([0-9]+) ([0-9]+)'
        if [[ "$line" =~ $pat ]]; then
            printf "  IP=%-15s  Time=%-26s  Method=%-6s  Path=%-20s  Status=%s  Size=%s\n" \
                "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" \
                "${BASH_REMATCH[4]}" "${BASH_REMATCH[5]}" "${BASH_REMATCH[6]}"
        fi
    }

    # Parse syslog line
    parse_syslog() {
        local line="$1"
        local pat='^([0-9-]+ [0-9:]+) \[([A-Z]+)\] (.+)$'
        if [[ "$line" =~ $pat ]]; then
            printf "  Time=%-20s  Level=%-7s  Msg=%s\n" \
                "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
        fi
    }

    # Parse all lines
    echo "--- Parsed Entries ---"
    while IFS= read -r line; do
        local fmt
        fmt=$(detect_format "$line")
        case "$fmt" in
            apache)  parse_apache "$line" ;;
            syslog)  parse_syslog "$line" ;;
            *)       echo "  [UNKNOWN] $line" ;;
        esac
    done < "$log_file"

    # Statistics
    echo ""
    echo "--- Statistics ---"

    # HTTP status code counts (Apache lines only)
    echo "  HTTP Status Codes:"
    local pat='" ([0-9]{3}) '
    while IFS= read -r line; do
        if [[ "$line" =~ $pat ]]; then
            echo "${BASH_REMATCH[1]}"
        fi
    done < "$log_file" | sort | uniq -c | while read -r count code; do
        printf "    %s : %d\n" "$code" "$count"
    done

    # Top IPs
    echo "  Top IPs:"
    local ip_pat='^([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)'
    while IFS= read -r line; do
        if [[ "$line" =~ $ip_pat ]]; then
            echo "${BASH_REMATCH[1]}"
        fi
    done < "$log_file" | sort | uniq -c | sort -rn | while read -r count ip; do
        printf "    %-15s : %d requests\n" "$ip" "$count"
    done

    rm -f "$log_file"
}

# === Exercise 3: Data Sanitizer ===
# Problem: Remove/escape special characters, validate/normalize phone numbers,
# redact sensitive data, and validate URLs.
exercise_3() {
    echo "=== Exercise 3: Data Sanitizer ==="

    # Escape HTML special characters
    sanitize_html() {
        local str="$1"
        str="${str//&/&amp;}"
        str="${str//</&lt;}"
        str="${str//>/&gt;}"
        str="${str//\"/&quot;}"
        str="${str//\'/&#39;}"
        echo "$str"
    }

    # Escape for shell
    sanitize_shell() {
        local str="$1"
        # Remove characters that could cause command injection
        str=$(echo "$str" | tr -cd '[:alnum:] ._-/')
        echo "$str"
    }

    # Normalize phone number to (XXX) XXX-XXXX
    normalize_phone() {
        local phone="$1"
        # Extract digits only
        local digits
        digits=$(echo "$phone" | tr -cd '0-9')

        # Remove leading 1 if 11 digits
        if (( ${#digits} == 11 )) && [[ "$digits" == 1* ]]; then
            digits="${digits:1}"
        fi

        if (( ${#digits} == 10 )); then
            printf "(%s) %s-%s" "${digits:0:3}" "${digits:3:3}" "${digits:6:4}"
        else
            echo "[INVALID: $phone]"
        fi
    }

    # Redact sensitive data using regex
    redact_sensitive() {
        local text="$1"

        # Redact SSN: XXX-XX-XXXX
        local ssn_pat='[0-9]{3}-[0-9]{2}-[0-9]{4}'
        text=$(echo "$text" | sed -E "s/$ssn_pat/***-**-****/g")

        # Redact credit card-like numbers (4 groups of 4 digits)
        local cc_pat='[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}'
        text=$(echo "$text" | sed -E "s/$cc_pat/****-****-****-****/g")

        # Redact email addresses
        local email_pat='[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        text=$(echo "$text" | sed -E "s/$email_pat/[REDACTED_EMAIL]/g")

        echo "$text"
    }

    # Validate URL
    validate_url() {
        local url="$1"
        local pat='^(https?|ftp)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[^[:space:]]*)?$'
        if [[ "$url" =~ $pat ]]; then
            echo "  VALID:   $url"
            echo "    Protocol: ${BASH_REMATCH[1]}"
        else
            echo "  INVALID: $url"
        fi
    }

    # Test HTML sanitization
    echo "--- HTML Sanitization ---"
    echo "  Input:  <script>alert('xss')</script>"
    echo "  Output: $(sanitize_html "<script>alert('xss')</script>")"

    echo ""
    echo "--- Shell Sanitization ---"
    echo "  Input:  file.txt; rm -rf /"
    echo "  Output: $(sanitize_shell "file.txt; rm -rf /")"

    echo ""
    echo "--- Phone Normalization ---"
    for phone in "5551234567" "1-555-123-4567" "(555) 123-4567" "12345"; do
        printf "  %-20s -> %s\n" "$phone" "$(normalize_phone "$phone")"
    done

    echo ""
    echo "--- Redact Sensitive Data ---"
    local text="SSN: 123-45-6789, CC: 4111-1111-1111-1111, Email: user@example.com"
    echo "  Input:  $text"
    echo "  Output: $(redact_sensitive "$text")"

    echo ""
    echo "--- URL Validation ---"
    validate_url "https://example.com/path"
    validate_url "ftp://files.example.com"
    validate_url "not-a-url"
}

# === Exercise 4: Configuration File Parser ===
# Problem: Parse INI/key=value formats, support sections and types,
# validate values against patterns, and support variable substitution.
exercise_4() {
    echo "=== Exercise 4: Configuration File Parser ==="

    local config_file="/tmp/cfgparse_$$.ini"
    cat > "$config_file" << 'EOF'
# Application Configuration

[server]
host = localhost
port = 8080
workers = 4
debug = true

[database]
host = db.${server.host}
port = 5432
name = myapp
max_connections = 100

[logging]
level = INFO
file = /var/log/app.log
EOF

    declare -A config_data
    declare -A config_types

    parse_config() {
        local file="$1"
        local section=""

        while IFS= read -r line; do
            # Skip empty lines and comments
            [[ "$line" =~ ^[[:space:]]*$ ]] && continue
            [[ "$line" =~ ^[[:space:]]*[#\;] ]] && continue

            # Trim whitespace
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"

            # Section header
            if [[ "$line" =~ ^\[([a-zA-Z_][a-zA-Z0-9_]*)\]$ ]]; then
                section="${BASH_REMATCH[1]}"
                continue
            fi

            # Key = value
            if [[ "$line" =~ ^([a-zA-Z_][a-zA-Z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
                local key="${BASH_REMATCH[1]}"
                local value="${BASH_REMATCH[2]}"
                local full_key="${section}.${key}"

                # Detect type
                if [[ "$value" =~ ^-?[0-9]+$ ]]; then
                    config_types["$full_key"]="int"
                elif [[ "$value" =~ ^(true|false)$ ]]; then
                    config_types["$full_key"]="bool"
                else
                    config_types["$full_key"]="string"
                fi

                config_data["$full_key"]="$value"
            fi
        done < "$file"
    }

    # Variable substitution: ${section.key}
    resolve_variables() {
        for key in "${!config_data[@]}"; do
            local value="${config_data[$key]}"
            local ref_pat='\$\{([a-zA-Z_][a-zA-Z0-9_.]*)\}'
            while [[ "$value" =~ $ref_pat ]]; do
                local ref="${BASH_REMATCH[1]}"
                local replacement="${config_data[$ref]:-}"
                value="${value/\$\{$ref\}/$replacement}"
            done
            config_data["$key"]="$value"
        done
    }

    # Validate port values
    validate_config() {
        local errors=0
        for key in "${!config_data[@]}"; do
            if [[ "$key" =~ \.port$ ]]; then
                local value="${config_data[$key]}"
                if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1 || value > 65535 )); then
                    echo "  [ERROR] $key: invalid port '$value'"
                    (( errors++ ))
                fi
            fi
        done
        return $errors
    }

    echo "--- Original INI ---"
    cat "$config_file" | sed 's/^/  /'

    echo ""
    echo "--- Parsed Configuration ---"
    parse_config "$config_file"
    resolve_variables

    for key in $(echo "${!config_data[@]}" | tr ' ' '\n' | sort); do
        printf "  %-30s = %-20s [%s]\n" "$key" "${config_data[$key]}" "${config_types[$key]}"
    done

    echo ""
    echo "--- Validation ---"
    if validate_config; then
        echo "  [OK] All values valid"
    fi

    rm -f "$config_file"
}

# === Exercise 5: Pattern-Based Router ===
# Problem: Define routes with patterns, extract path parameters using
# BASH_REMATCH, support optional params, match by priority.
exercise_5() {
    echo "=== Exercise 5: Pattern-Based Router ==="

    # Route definitions: pattern -> handler name
    declare -a route_patterns
    declare -a route_handlers
    declare -a route_param_names

    add_route() {
        local pattern="$1"
        local handler="$2"
        local param_names="$3"

        route_patterns+=("$pattern")
        route_handlers+=("$handler")
        route_param_names+=("$param_names")
    }

    # Convert route pattern like /user/:id to regex ^/user/([^/]+)$
    compile_route() {
        local route="$1"
        local regex="$route"

        # Replace :param([0-9]+) with constrained capture
        regex=$(echo "$regex" | sed -E 's/:([a-zA-Z_]+)\(([^)]+)\)/(\2)/g')
        # Replace :param? with optional capture
        regex=$(echo "$regex" | sed -E 's/:([a-zA-Z_]+)\?/([^\/]*)?/g')
        # Replace :param with generic capture
        regex=$(echo "$regex" | sed -E 's/:([a-zA-Z_]+)/([^\/]+)/g')

        echo "^${regex}$"
    }

    # Extract parameter names from route pattern
    extract_param_names() {
        local route="$1"
        local names=""
        local pat=':([a-zA-Z_]+)'
        local tmp="$route"

        while [[ "$tmp" =~ $pat ]]; do
            [ -n "$names" ] && names+=" "
            names+="${BASH_REMATCH[1]}"
            tmp="${tmp#*${BASH_REMATCH[0]}}"
        done
        echo "$names"
    }

    # Register routes
    add_route "/users" "list_users" ""
    add_route "/user/([0-9]+)" "get_user" "id"
    add_route "/posts/([0-9]{4})/([0-9]{2})/([^/]+)" "get_post" "year month slug"
    add_route "/files/(.*)" "get_file" "path"

    # Match a URL against registered routes
    match_route() {
        local url="$1"

        # Parse query string
        local path="${url%%\?*}"
        local query=""
        [[ "$url" == *"?"* ]] && query="${url#*\?}"

        for i in "${!route_patterns[@]}"; do
            local pattern="^${route_patterns[$i]}$"
            if [[ "$path" =~ $pattern ]]; then
                echo "  Matched: ${route_handlers[$i]}"
                echo "  Pattern: ${route_patterns[$i]}"

                # Extract parameters
                local IFS=' '
                local names=(${route_param_names[$i]})
                for j in "${!names[@]}"; do
                    local idx=$(( j + 1 ))
                    echo "    ${names[$j]} = ${BASH_REMATCH[$idx]}"
                done

                # Parse query string
                if [ -n "$query" ]; then
                    echo "  Query parameters:"
                    IFS='&' read -ra pairs <<< "$query"
                    for pair in "${pairs[@]}"; do
                        local key="${pair%%=*}"
                        local value="${pair#*=}"
                        echo "    $key = $value"
                    done
                fi

                return 0
            fi
        done

        echo "  No route matched for: $url"
        return 1
    }

    # Generate URL from pattern and parameters
    generate_url() {
        local pattern="$1"
        shift
        local result="$pattern"

        while [ $# -gt 0 ]; do
            local key="${1%%=*}"
            local value="${1#*=}"
            # Replace the first capture group placeholder with value
            result=$(echo "$result" | sed -E "s/\([^)]+\)/$value/")
            shift
        done

        echo "$result"
    }

    # Test routing
    echo "--- Route Matching ---"
    echo "URL: /users"
    match_route "/users"
    echo ""

    echo "URL: /user/42"
    match_route "/user/42"
    echo ""

    echo "URL: /posts/2024/02/hello-world"
    match_route "/posts/2024/02/hello-world"
    echo ""

    echo "URL: /files/docs/readme.txt"
    match_route "/files/docs/readme.txt"
    echo ""

    echo "URL: /user/42?format=json&verbose=true"
    match_route "/user/42?format=json&verbose=true"
    echo ""

    echo "URL: /unknown"
    match_route "/unknown"
    echo ""

    echo "--- URL Generation ---"
    echo "  $(generate_url "/user/([0-9]+)" "id=99")"
    echo "  $(generate_url "/posts/([0-9]{4})/([0-9]{2})/([^/]+)" "year=2024" "month=03" "slug=test")"
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
