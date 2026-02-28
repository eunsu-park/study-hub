#!/bin/bash
# Exercises for Lesson 07: String Processing and Text Manipulation
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Advanced CSV Processor ===
# Problem: Read CSV with header, validate rows, filter, sort, and
# output in CSV, JSON, or formatted table.
exercise_1() {
    echo "=== Exercise 1: Advanced CSV Processor ==="

    local csv_file="/tmp/csvproc_$$.csv"
    cat > "$csv_file" << 'EOF'
name,age,city,salary
Alice,30,New York,80000
Bob,25,Los Angeles,65000
Charlie,35,Chicago,90000
Diana,28,New York,72000
Eve,42,Chicago,95000
Frank,22,Boston,55000
EOF

    # Read header
    local header
    IFS= read -r header < "$csv_file"
    IFS=',' read -ra cols <<< "$header"

    echo "Columns: ${cols[*]}"
    echo ""

    # Output as formatted table
    echo "--- Formatted Table ---"
    printf "%-12s %5s %-15s %10s\n" "${cols[@]}"
    printf "%-12s %5s %-15s %10s\n" "------------" "-----" "---------------" "----------"
    tail -n +2 "$csv_file" | while IFS=',' read -ra fields; do
        printf "%-12s %5s %-15s %10s\n" "${fields[@]}"
    done

    # Filter: age > 28
    echo ""
    echo "--- Filter: age > 28 ---"
    printf "%-12s %5s %-15s %10s\n" "${cols[@]}"
    tail -n +2 "$csv_file" | while IFS=',' read -r name age city salary; do
        if (( age > 28 )); then
            printf "%-12s %5s %-15s %10s\n" "$name" "$age" "$city" "$salary"
        fi
    done

    # Sort by salary descending
    echo ""
    echo "--- Sorted by salary (desc) ---"
    printf "%-12s %5s %-15s %10s\n" "${cols[@]}"
    tail -n +2 "$csv_file" | sort -t',' -k4 -rn | while IFS=',' read -ra fields; do
        printf "%-12s %5s %-15s %10s\n" "${fields[@]}"
    done

    # Output as JSON
    echo ""
    echo "--- JSON Output ---"
    echo "["
    local first=true
    tail -n +2 "$csv_file" | while IFS=',' read -r name age city salary; do
        if $first; then
            first=false
        else
            echo ","
        fi
        printf '  {"name": "%s", "age": %s, "city": "%s", "salary": %s}' \
            "$name" "$age" "$city" "$salary"
    done
    echo ""
    echo "]"

    rm -f "$csv_file"
}

# === Exercise 2: Log Parser and Analyzer ===
# Problem: Parse log files, extract structured data, generate statistics
# and a timeline visualization.
exercise_2() {
    echo "=== Exercise 2: Log Parser and Analyzer ==="

    local log_file="/tmp/logparse_$$.log"
    cat > "$log_file" << 'EOF'
2024-02-13 10:00:01 [INFO] Server started on port 8080
2024-02-13 10:05:22 [WARN] High memory usage: 82%
2024-02-13 10:10:45 [ERROR] Database connection lost
2024-02-13 10:11:00 [INFO] Reconnecting to database...
2024-02-13 10:11:05 [INFO] Database reconnected
2024-02-13 10:15:30 [ERROR] Request timeout: /api/users
2024-02-13 10:20:00 [INFO] Health check passed
2024-02-13 10:25:10 [WARN] Disk usage at 90%
2024-02-13 10:30:00 [ERROR] Out of memory
2024-02-13 10:30:05 [INFO] Emergency cleanup triggered
2024-02-13 10:35:00 [INFO] System recovered
EOF

    # Parse and extract structured data
    echo "--- Parsed Log Entries ---"
    printf "%-20s %-7s %s\n" "Timestamp" "Level" "Message"
    printf "%-20s %-7s %s\n" "--------------------" "-------" "----------------------------"

    while IFS= read -r line; do
        if [[ "$line" =~ ^([0-9-]+\ [0-9:]+)\ \[([A-Z]+)\]\ (.+)$ ]]; then
            printf "%-20s %-7s %s\n" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
        fi
    done < "$log_file"

    # Statistics
    echo ""
    echo "--- Statistics ---"
    for level in INFO WARN ERROR; do
        local count
        count=$(grep -c "\[$level\]" "$log_file")
        printf "  %-7s : %d\n" "$level" "$count"
    done

    local total
    total=$(wc -l < "$log_file")
    local errors
    errors=$(grep -c "\[ERROR\]" "$log_file")
    echo "  Total  : $total"
    echo "  Error rate: $(( errors * 100 / total ))%"

    # Top errors
    echo ""
    echo "--- Error Messages ---"
    grep "\[ERROR\]" "$log_file" | sed 's/.*\[ERROR\] /  - /'

    # ASCII timeline (errors per 10-minute window)
    echo ""
    echo "--- Timeline (events per 10-min window) ---"
    grep -oP '\d{2}:\d{2}' "$log_file" | cut -c1-4 | sort | uniq -c | while read -r count window; do
        local bar=""
        for (( i=0; i<count; i++ )); do bar+="#"; done
        printf "  %s0 : %-10s (%d)\n" "$window" "$bar" "$count"
    done

    rm -f "$log_file"
}

# === Exercise 3: Configuration File Converter ===
# Problem: Convert between INI and JSON-like formats.
exercise_3() {
    echo "=== Exercise 3: Configuration File Converter ==="

    local ini_file="/tmp/cfgconv_$$.ini"
    cat > "$ini_file" << 'EOF'
[server]
host = localhost
port = 8080
workers = 4

[database]
host = db.example.com
port = 5432
name = myapp
EOF

    # INI to JSON conversion
    ini_to_json() {
        local file="$1"
        local section=""
        local first_section=true
        local first_key=true

        echo "{"
        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*$ ]] && continue
            [[ "$line" =~ ^[[:space:]]*[#\;] ]] && continue

            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"

            if [[ "$line" =~ ^\[(.+)\]$ ]]; then
                if [ -n "$section" ]; then
                    echo ""
                    echo "  },"
                fi
                section="${BASH_REMATCH[1]}"
                first_key=true
                if $first_section; then
                    first_section=false
                fi
                printf '  "%s": {\n' "$section"
            elif [[ "$line" == *=* ]]; then
                local key="${line%%=*}"
                local value="${line#*=}"
                key="${key%"${key##*[![:space:]]}"}"
                value="${value#"${value%%[![:space:]]*}"}"

                if $first_key; then
                    first_key=false
                else
                    echo ","
                fi

                # Detect type
                if [[ "$value" =~ ^[0-9]+$ ]]; then
                    printf '    "%s": %s' "$key" "$value"
                elif [[ "$value" =~ ^(true|false)$ ]]; then
                    printf '    "%s": %s' "$key" "$value"
                else
                    printf '    "%s": "%s"' "$key" "$value"
                fi
            fi
        done < "$file"

        if [ -n "$section" ]; then
            echo ""
            echo "  }"
        fi
        echo "}"
    }

    # INI to ENV conversion
    ini_to_env() {
        local file="$1"
        local section=""

        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*$ ]] && continue
            [[ "$line" =~ ^[[:space:]]*[#\;] ]] && continue

            line="${line#"${line%%[![:space:]]*}"}"

            if [[ "$line" =~ ^\[(.+)\]$ ]]; then
                section="${BASH_REMATCH[1]^^}"
            elif [[ "$line" == *=* ]]; then
                local key="${line%%=*}"
                local value="${line#*=}"
                key="${key%"${key##*[![:space:]]}"}"
                value="${value#"${value%%[![:space:]]*}"}"
                echo "${section}_${key^^}=$value"
            fi
        done < "$file"
    }

    echo "--- Original INI ---"
    cat "$ini_file"

    echo ""
    echo "--- Converted to JSON ---"
    ini_to_json "$ini_file"

    echo ""
    echo "--- Converted to ENV ---"
    ini_to_env "$ini_file"

    rm -f "$ini_file"
}

# === Exercise 4: Text Template Engine ===
# Problem: Process templates with {{variable}} placeholders and basic
# conditional blocks.
exercise_4() {
    echo "=== Exercise 4: Text Template Engine ==="

    render_template() {
        local template="$1"
        shift

        # Build associative array of variables
        declare -A vars
        for arg in "$@"; do
            local key="${arg%%=*}"
            local value="${arg#*=}"
            vars["$key"]="$value"
        done

        local result="$template"

        # Process {{#if var}}...{{/if}} blocks
        while [[ "$result" =~ \{\{#if\ ([a-zA-Z_]+)\}\}([^{]*)\{\{/if\}\} ]]; do
            local var="${BASH_REMATCH[1]}"
            local body="${BASH_REMATCH[2]}"
            if [ -n "${vars[$var]:-}" ] && [ "${vars[$var]}" != "false" ]; then
                result="${result//"{{#if $var}}${body}{{/if}}"/$body}"
            else
                result="${result//"{{#if $var}}${body}{{/if}}"/}"
            fi
        done

        # Apply filters: {{var|upper}} and {{var|lower}}
        for key in "${!vars[@]}"; do
            local upper="${vars[$key]^^}"
            local lower="${vars[$key],,}"
            result="${result//"{{${key}|upper}}"/$upper}"
            result="${result//"{{${key}|lower}}"/$lower}"
            result="${result//"{{$key}}"/${vars[$key]}}"
        done

        echo "$result"
    }

    echo "Test 1: Simple substitution"
    render_template "Hello, {{name}}! Welcome to {{city}}." name="Alice" city="New York"

    echo ""
    echo "Test 2: Conditional (true)"
    render_template "Hello!{{#if admin}} You have admin access.{{/if}}" admin="true"

    echo ""
    echo "Test 3: Conditional (false)"
    render_template "Hello!{{#if admin}} You have admin access.{{/if}}" admin=""

    echo ""
    echo "Test 4: Filters"
    render_template "Name: {{name|upper}}, City: {{city|lower}}" name="alice" city="NEW YORK"
}

# === Exercise 5: Data Validation Framework ===
# Problem: Define validation rules and validate data against them,
# reporting errors with field names.
exercise_5() {
    echo "=== Exercise 5: Data Validation Framework ==="

    validate_field() {
        local field_name="$1"
        local value="$2"
        local rule="$3"
        local param="${4:-}"

        case "$rule" in
            required)
                if [ -z "$value" ]; then
                    echo "  [FAIL] $field_name: required field is empty"
                    return 1
                fi
                ;;
            type:int)
                if [[ ! "$value" =~ ^-?[0-9]+$ ]]; then
                    echo "  [FAIL] $field_name: '$value' is not an integer"
                    return 1
                fi
                ;;
            type:email)
                if [[ ! "$value" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
                    echo "  [FAIL] $field_name: '$value' is not a valid email"
                    return 1
                fi
                ;;
            range)
                local min="${param%-*}"
                local max="${param#*-}"
                if (( value < min || value > max )); then
                    echo "  [FAIL] $field_name: $value not in range $min-$max"
                    return 1
                fi
                ;;
            length)
                local min="${param%-*}"
                local max="${param#*-}"
                local len=${#value}
                if (( len < min || len > max )); then
                    echo "  [FAIL] $field_name: length $len not in $min-$max"
                    return 1
                fi
                ;;
            pattern)
                if [[ ! "$value" =~ $param ]]; then
                    echo "  [FAIL] $field_name: '$value' doesn't match pattern"
                    return 1
                fi
                ;;
        esac
        return 0
    }

    validate_record() {
        local errors=0

        # Validate each field against multiple rules
        for rule_spec in "$@"; do
            local field="${rule_spec%%:*}"
            local rest="${rule_spec#*:}"
            local value="${rest%%:*}"
            local rule="${rest#*:}"
            local param="${rule#*:}"
            [ "$param" = "$rule" ] && param=""

            if ! validate_field "$field" "$value" "$rule" "$param"; then
                (( errors++ ))
            fi
        done

        return $errors
    }

    echo "--- Validating Record 1 (valid) ---"
    local err=0
    validate_field "name"   "Alice"              "required" || (( err++ ))
    validate_field "name"   "Alice"              "length" "1-50" || (( err++ ))
    validate_field "age"    "30"                 "type:int" || (( err++ ))
    validate_field "age"    "30"                 "range" "0-150" || (( err++ ))
    validate_field "email"  "alice@example.com"  "type:email" || (( err++ ))
    if (( err == 0 )); then
        echo "  [OK] Record is valid"
    else
        echo "  Total errors: $err"
    fi

    echo ""
    echo "--- Validating Record 2 (invalid) ---"
    err=0
    validate_field "name"   ""                   "required" || (( err++ ))
    validate_field "age"    "abc"                "type:int" || (( err++ ))
    validate_field "age"    "200"                "range" "0-150" || (( err++ ))
    validate_field "email"  "not-an-email"       "type:email" || (( err++ ))
    validate_field "code"   "AB"                 "length" "3-10" || (( err++ ))
    echo "  Total errors: $err"
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
