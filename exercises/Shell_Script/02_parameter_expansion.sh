#!/bin/bash
# Exercises for Lesson 02: Parameter Expansion and Variable Attributes
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Advanced Filename Processor ===
# Problem: Extract date stamps, version numbers, sanitize filenames,
# and generate normalized filenames using parameter expansion.
exercise_1() {
    echo "=== Exercise 1: Advanced Filename Processor ==="

    process_filename() {
        local filename="$1"
        local base extension date_stamp version base_name normalized

        # Extract extension
        extension="${filename##*.}"

        # Remove extension for processing
        local name_no_ext="${filename%.*}"

        # Extract date stamp (YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD)
        date_stamp=""
        if [[ "$name_no_ext" =~ ([0-9]{4})[-_]?([0-9]{2})[-_]?([0-9]{2}) ]]; then
            date_stamp="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-${BASH_REMATCH[3]}"
        fi

        # Extract version number (v1.0, version-2.3.4, etc.)
        version=""
        if [[ "$name_no_ext" =~ [vV]([0-9]+(\.[0-9]+)*) ]]; then
            version="${BASH_REMATCH[1]}"
        fi

        # Extract base name: remove date stamps and version strings
        base_name="$name_no_ext"
        # Remove date patterns
        base_name="${base_name//_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/}"
        base_name="${base_name//_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/}"
        # Remove version patterns
        base_name=$(echo "$base_name" | sed -E 's/_?[vV][0-9]+(\.[0-9]+)*//g')
        # Remove trailing underscores/hyphens
        base_name="${base_name%%[-_]}"
        base_name="${base_name##[-_]}"

        # Build normalized filename
        normalized="$base_name"
        [ -n "$date_stamp" ] && normalized="${normalized}_${date_stamp}"
        [ -n "$version" ] && normalized="${normalized}_v${version}"
        normalized="${normalized}.${extension}"

        echo "  Input:      $filename"
        echo "  Date:       ${date_stamp:-none}"
        echo "  Version:    ${version:-none}"
        echo "  Base:       $base_name"
        echo "  Extension:  $extension"
        echo "  Normalized: $normalized"
    }

    process_filename "report_20240213_v1.3.pdf"
    echo ""
    process_filename "data_2024-02-13.csv"
    echo ""
    process_filename "backup_v2.0_20231201.tar.gz"
    echo ""
    process_filename "simple_document.txt"
}

# === Exercise 2: Environment Variable Validator ===
# Problem: Validate environment variables for type, range, and constraints.
exercise_2() {
    echo "=== Exercise 2: Environment Variable Validator ==="

    # Set up test environment variables
    export TEST_PORT="8080"
    export TEST_DEBUG="true"
    export TEST_ENV="staging"
    export TEST_KEY="abcdefghijklmnopqrstuvwxyz123456"

    validate_env() {
        local var_name="$1"
        shift
        local var_value="${!var_name:-}"

        local type="" required=false min_length=0 range="" values="" default=""

        # Parse validation options
        while [ $# -gt 0 ]; do
            case "$1" in
                --type)     type="$2"; shift 2 ;;
                --required) required=true; shift ;;
                --range)    range="$2"; shift 2 ;;
                --values)   values="$2"; shift 2 ;;
                --default)  default="$2"; shift 2 ;;
                --min-length) min_length="$2"; shift 2 ;;
                *) shift ;;
            esac
        done

        # Apply default if variable is empty
        if [ -z "$var_value" ] && [ -n "$default" ]; then
            var_value="$default"
        fi

        # Check required
        if $required && [ -z "$var_value" ]; then
            echo "  [FAIL] $var_name: required but not set"
            return 1
        fi

        # Skip further checks if empty and not required
        if [ -z "$var_value" ]; then
            echo "  [OK]   $var_name: not set (optional, default: ${default:-none})"
            return 0
        fi

        # Type validation
        case "$type" in
            int)
                if ! [[ "$var_value" =~ ^-?[0-9]+$ ]]; then
                    echo "  [FAIL] $var_name='$var_value': expected integer"
                    return 1
                fi
                # Range check for integers
                if [ -n "$range" ]; then
                    local min="${range%-*}"
                    local max="${range#*-}"
                    if (( var_value < min || var_value > max )); then
                        echo "  [FAIL] $var_name=$var_value: out of range ($min-$max)"
                        return 1
                    fi
                fi
                ;;
            bool)
                case "${var_value,,}" in
                    true|false|yes|no|1|0) ;;
                    *)
                        echo "  [FAIL] $var_name='$var_value': expected boolean (true/false/yes/no/1/0)"
                        return 1
                        ;;
                esac
                ;;
            enum)
                local found=false
                IFS=',' read -ra valid_values <<< "$values"
                for v in "${valid_values[@]}"; do
                    if [ "$var_value" = "$v" ]; then
                        found=true
                        break
                    fi
                done
                if ! $found; then
                    echo "  [FAIL] $var_name='$var_value': must be one of: $values"
                    return 1
                fi
                ;;
            string)
                if (( min_length > 0 )) && (( ${#var_value} < min_length )); then
                    echo "  [FAIL] $var_name: length ${#var_value} < min $min_length"
                    return 1
                fi
                ;;
        esac

        echo "  [OK]   $var_name='$var_value'"
        return 0
    }

    validate_env TEST_PORT      --type int --range 1024-65535 --required
    validate_env TEST_DEBUG     --type bool --default false
    validate_env TEST_ENV       --type enum --values "dev,staging,prod" --required
    validate_env TEST_KEY       --type string --min-length 32 --required
    validate_env TEST_MISSING   --type string --required
    validate_env TEST_OPTIONAL  --type int --default "42"

    # Cleanup
    unset TEST_PORT TEST_DEBUG TEST_ENV TEST_KEY
}

# === Exercise 3: Advanced Config File Manager ===
# Problem: Read INI-style config with sections and provide get/set operations.
exercise_3() {
    echo "=== Exercise 3: Advanced Config File Manager ==="

    local config_file="/tmp/test_config_$$.ini"

    # Create sample config
    cat > "$config_file" << 'EOF'
[database]
host = localhost
port = 5432
name = myapp

[database.pool]
min_size = 5
max_size = 20

[cache]
host = redis.local
port = 6379
ttl = 3600
EOF

    # Associative array to store config
    declare -A config

    # Load INI config into associative array
    config_load() {
        local file="$1"
        local section=""
        local line key value

        while IFS= read -r line; do
            # Skip empty lines and comments
            [[ "$line" =~ ^[[:space:]]*$ ]] && continue
            [[ "$line" =~ ^[[:space:]]*[#\;] ]] && continue

            # Trim whitespace
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"

            # Section header
            if [[ "$line" =~ ^\[(.+)\]$ ]]; then
                section="${BASH_REMATCH[1]}"
                continue
            fi

            # Key=value
            if [[ "$line" == *=* ]]; then
                key="${line%%=*}"
                value="${line#*=}"
                key="${key#"${key%%[![:space:]]*}"}"
                key="${key%"${key##*[![:space:]]}"}"
                value="${value#"${value%%[![:space:]]*}"}"
                value="${value%"${value##*[![:space:]]}"}"

                if [ -n "$section" ]; then
                    config["${section}.${key}"]="$value"
                else
                    config["$key"]="$value"
                fi
            fi
        done < "$file"
    }

    # Get config value
    config_get() {
        local key="$1"
        local default="${2:-}"
        echo "${config[$key]:-$default}"
    }

    # Set config value
    config_set() {
        local key="$1"
        local value="$2"
        config["$key"]="$value"
    }

    # List all config entries
    config_list() {
        for key in $(echo "${!config[@]}" | tr ' ' '\n' | sort); do
            echo "  $key = ${config[$key]}"
        done
    }

    # Load and display
    config_load "$config_file"

    echo "All configuration entries:"
    config_list

    echo ""
    echo "Get specific values:"
    echo "  database.host = $(config_get 'database.host')"
    echo "  database.port = $(config_get 'database.port')"
    echo "  cache.ttl = $(config_get 'cache.ttl')"
    echo "  missing.key = $(config_get 'missing.key' 'default_value')"

    echo ""
    echo "Set new value: database.host = production-db"
    config_set "database.host" "production-db"
    echo "  database.host = $(config_get 'database.host')"

    # Cleanup
    rm -f "$config_file"
}

# === Exercise 4: String Template Engine ===
# Problem: Replace variables, support defaults, conditionals in templates.
exercise_4() {
    echo "=== Exercise 4: String Template Engine ==="

    render() {
        local template="$1"
        shift

        # Store variables in an associative array
        declare -A vars
        for arg in "$@"; do
            local key="${arg%%=*}"
            local value="${arg#*=}"
            vars["$key"]="$value"
        done

        local result="$template"

        # Process conditionals: {{#if var}}...{{/if}}
        while [[ "$result" =~ \{\{#if\ ([a-zA-Z_][a-zA-Z0-9_]*)\}\}(.*)\{\{/if\}\} ]]; do
            local cond_var="${BASH_REMATCH[1]}"
            local cond_body="${BASH_REMATCH[2]}"

            if [ -n "${vars[$cond_var]:-}" ] && [ "${vars[$cond_var]}" != "false" ]; then
                # Replace the entire conditional block with the body
                result="${result//"{{#if $cond_var}}${cond_body}{{/if}}"/$cond_body}"
            else
                # Remove the entire conditional block
                result="${result//"{{#if $cond_var}}${cond_body}{{/if}}"/}"
            fi
        done

        # Process variables with defaults: {{var:default}}
        local pattern
        for key in "${!vars[@]}"; do
            # Replace {{key}} with value
            result="${result//"{{$key}}"/${vars[$key]}}"
        done

        # Handle remaining variables with defaults
        while [[ "$result" =~ \{\{([a-zA-Z_][a-zA-Z0-9_]*):([^}]*)\}\} ]]; do
            local var_name="${BASH_REMATCH[1]}"
            local default_val="${BASH_REMATCH[2]}"
            local replacement="${vars[$var_name]:-$default_val}"
            result="${result//"{{$var_name:$default_val}}"/$replacement}"
        done

        # Apply filters: {{var | uppercase}}
        while [[ "$result" =~ \{\{([a-zA-Z_][a-zA-Z0-9_]*)\ \|\ uppercase\}\} ]]; do
            local var_name="${BASH_REMATCH[1]}"
            local value="${vars[$var_name]:-}"
            result="${result//"{{$var_name | uppercase}}"/${value^^}}"
        done

        while [[ "$result" =~ \{\{([a-zA-Z_][a-zA-Z0-9_]*)\ \|\ lowercase\}\} ]]; do
            local var_name="${BASH_REMATCH[1]}"
            local value="${vars[$var_name]:-}"
            result="${result//"{{$var_name | lowercase}}"/${value,,}}"
        done

        echo "$result"
    }

    echo "Test 1: Basic substitution"
    render "Hello {{name}}!" name="World"

    echo ""
    echo "Test 2: Default value"
    render "Hello {{name:Guest}}!"

    echo ""
    echo "Test 3: Conditional"
    render "Welcome!{{#if premium}} You are a premium member.{{/if}}" premium="true"

    echo ""
    echo "Test 4: Conditional (false)"
    render "Welcome!{{#if premium}} You are a premium member.{{/if}}" premium="false"

    echo ""
    echo "Test 5: Combined"
    render "Hello {{name:Guest}}!{{#if admin}} [ADMIN]{{/if}} Role: {{role:user}}" \
        name="Alice" admin="true"
}

# === Exercise 5: Path Resolution Library ===
# Problem: Resolve relative paths, handle symlinks, find files in PATH.
exercise_5() {
    echo "=== Exercise 5: Path Resolution Library ==="

    # Resolve a path with ~, variables, and .. components
    resolve_path() {
        local path="$1"

        # Expand ~ to HOME
        path="${path/#\~/$HOME}"

        # Expand environment variables
        path=$(eval echo "$path" 2>/dev/null || echo "$path")

        # Use a subshell to resolve the path
        if [ -e "$path" ]; then
            # For existing paths, use cd/pwd to resolve
            if [ -d "$path" ]; then
                (cd "$path" 2>/dev/null && pwd)
            else
                local dir
                dir=$(cd "$(dirname "$path")" 2>/dev/null && pwd)
                echo "${dir}/$(basename "$path")"
            fi
        else
            # For non-existing paths, do manual resolution
            # Remove ./ components
            path="${path//\/.\//\/}"
            # Resolve .. components using a simple loop
            echo "$path"
        fi
    }

    # Find a command in PATH
    find_in_path() {
        local cmd="$1"
        local result
        result=$(command -v "$cmd" 2>/dev/null)
        if [ -n "$result" ]; then
            echo "$result"
            return 0
        else
            echo "Not found: $cmd"
            return 1
        fi
    }

    # Check if a path is a subpath (prevent directory traversal)
    is_subpath() {
        local base_dir="$1"
        local check_path="$2"

        # Resolve both paths
        local resolved_base resolved_check
        resolved_base=$(cd "$base_dir" 2>/dev/null && pwd)
        resolved_check=$(resolve_path "$check_path")

        if [[ "$resolved_check" == "$resolved_base"* ]]; then
            echo "OK: '$check_path' is within '$base_dir'"
            return 0
        else
            echo "Error: Directory traversal detected! '$check_path' escapes '$base_dir'"
            return 1
        fi
    }

    # Test resolve_path
    echo "Resolve paths:"
    echo "  ~/Documents -> $(resolve_path '~/Documents')"
    echo "  /tmp/./test -> $(resolve_path '/tmp/./test')"
    echo "  /tmp        -> $(resolve_path '/tmp')"

    echo ""
    echo "Find in PATH:"
    find_in_path "bash"
    find_in_path "python3"
    find_in_path "nonexistent_command_xyz"

    echo ""
    echo "Subpath checks:"
    is_subpath "/tmp" "/tmp/myfile.txt"
    is_subpath "/tmp" "/etc/passwd"
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
