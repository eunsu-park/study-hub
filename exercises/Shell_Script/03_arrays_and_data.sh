#!/bin/bash
# Exercises for Lesson 03: Arrays and Data Structures
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Advanced CSV Processor ===
# Problem: Parse CSV, filter rows, sort by column, calculate aggregates.
exercise_1() {
    echo "=== Exercise 1: Advanced CSV Processor ==="

    # Create sample CSV data
    local csv_file="/tmp/employees_$$.csv"
    cat > "$csv_file" << 'EOF'
name,age,city,salary
Alice,30,New York,80000
Bob,25,Los Angeles,65000
Charlie,35,Chicago,90000
Diana,28,New York,72000
Eve,42,Chicago,95000
EOF

    echo "--- Original Data ---"
    cat "$csv_file"
    echo ""

    # Parse CSV header
    local header
    IFS= read -r header < "$csv_file"
    IFS=',' read -ra columns <<< "$header"

    # Find column index by name
    find_column() {
        local target="$1"
        for i in "${!columns[@]}"; do
            if [ "${columns[$i]}" = "$target" ]; then
                echo "$i"
                return 0
            fi
        done
        return 1
    }

    # Filter rows where age > 30
    echo "--- Filter: age > 30 ---"
    local age_col
    age_col=$(find_column "age")
    echo "$header"
    tail -n +2 "$csv_file" | while IFS=',' read -ra fields; do
        if (( fields[age_col] > 30 )); then
            echo "${fields[*]}" | tr ' ' ','
        fi
    done
    echo ""

    # Sort by salary (descending)
    echo "--- Sort by salary (descending) ---"
    echo "$header"
    tail -n +2 "$csv_file" | sort -t',' -k4 -rn
    echo ""

    # Aggregate: average salary
    echo "--- Aggregate: salary statistics ---"
    local sum=0 count=0 min=999999999 max=0
    local salary_col
    salary_col=$(find_column "salary")

    while IFS=',' read -ra fields; do
        local sal="${fields[$salary_col]}"
        (( sum += sal ))
        (( count++ ))
        (( sal < min )) && min=$sal
        (( sal > max )) && max=$sal
    done < <(tail -n +2 "$csv_file")

    local avg=$(( sum / count ))
    echo "  Count: $count"
    echo "  Sum:   $sum"
    echo "  Avg:   $avg"
    echo "  Min:   $min"
    echo "  Max:   $max"

    rm -f "$csv_file"
}

# === Exercise 2: Configuration Manager ===
# Problem: Load configs from multiple sources with layered precedence.
exercise_2() {
    echo "=== Exercise 2: Configuration Manager ==="

    # Create config files
    local defaults_file="/tmp/defaults_$$.conf"
    local user_file="/tmp/user_$$.conf"

    cat > "$defaults_file" << 'EOF'
host = localhost
port = 8080
debug = false
max_connections = 100
timeout = 30
EOF

    cat > "$user_file" << 'EOF'
host = production.example.com
debug = true
workers = 4
EOF

    declare -A config

    # Load a config file into the config array
    config_load() {
        local file="$1"
        [ -f "$file" ] || return 1

        local line key value
        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*$ ]] && continue
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ "$line" != *=* ]] && continue

            key="${line%%=*}"
            value="${line#*=}"
            key="${key#"${key%%[![:space:]]*}"}"
            key="${key%"${key##*[![:space:]]}"}"
            value="${value#"${value%%[![:space:]]*}"}"
            value="${value%"${value##*[![:space:]]}"}"

            config["$key"]="$value"
        done < "$file"
    }

    config_get() {
        local key="$1"
        local default="${2:-}"
        echo "${config[$key]:-$default}"
    }

    config_set() {
        config["$1"]="$2"
    }

    config_list() {
        for key in $(echo "${!config[@]}" | tr ' ' '\n' | sort); do
            printf "  %-20s = %s\n" "$key" "${config[$key]}"
        done
    }

    config_validate() {
        local errors=0
        # Check required fields
        for required in host port; do
            if [ -z "${config[$required]:-}" ]; then
                echo "  [ERROR] Missing required: $required"
                (( errors++ ))
            fi
        done

        # Type checking
        if [ -n "${config[port]:-}" ]; then
            if ! [[ "${config[port]}" =~ ^[0-9]+$ ]]; then
                echo "  [ERROR] 'port' must be numeric: ${config[port]}"
                (( errors++ ))
            fi
        fi

        if (( errors == 0 )); then
            echo "  [OK] Configuration is valid"
        fi
        return $errors
    }

    # Layer 1: Load defaults
    echo "Loading defaults..."
    config_load "$defaults_file"

    # Layer 2: Load user overrides
    echo "Loading user overrides..."
    config_load "$user_file"

    # Layer 3: Environment variable overrides
    [ -n "${APP_PORT:-}" ] && config_set "port" "$APP_PORT"

    echo ""
    echo "Final configuration:"
    config_list

    echo ""
    echo "Validation:"
    config_validate

    rm -f "$defaults_file" "$user_file"
}

# === Exercise 3: Data Structure Library ===
# Problem: Implement stack, queue, and set data structures.
exercise_3() {
    echo "=== Exercise 3: Data Structure Library ==="

    # --- Stack ---
    echo "--- Stack Demo ---"
    declare -a stack=()

    stack_push() { stack+=("$1"); }
    stack_pop() {
        if [ ${#stack[@]} -eq 0 ]; then echo ""; return 1; fi
        local val="${stack[-1]}"
        unset 'stack[-1]'
        echo "$val"
    }
    stack_peek() {
        if [ ${#stack[@]} -eq 0 ]; then echo ""; return 1; fi
        echo "${stack[-1]}"
    }
    stack_size() { echo "${#stack[@]}"; }

    stack_push "first"
    stack_push "second"
    stack_push "third"
    echo "  Push: first, second, third"
    echo "  Peek: $(stack_peek)"
    echo "  Size: $(stack_size)"
    echo "  Pop:  $(stack_pop)"
    echo "  Pop:  $(stack_pop)"
    echo "  Size: $(stack_size)"
    echo ""

    # --- Queue ---
    echo "--- Queue Demo ---"
    declare -a queue=()

    queue_enqueue() { queue+=("$1"); }
    queue_dequeue() {
        if [ ${#queue[@]} -eq 0 ]; then echo ""; return 1; fi
        local val="${queue[0]}"
        queue=("${queue[@]:1}")
        echo "$val"
    }
    queue_front() {
        if [ ${#queue[@]} -eq 0 ]; then echo ""; return 1; fi
        echo "${queue[0]}"
    }
    queue_size() { echo "${#queue[@]}"; }

    queue_enqueue "job1"
    queue_enqueue "job2"
    queue_enqueue "job3"
    echo "  Enqueue: job1, job2, job3"
    echo "  Front:   $(queue_front)"
    echo "  Dequeue: $(queue_dequeue)"
    echo "  Dequeue: $(queue_dequeue)"
    echo "  Front:   $(queue_front)"
    echo "  Size:    $(queue_size)"
    echo ""

    # --- Set ---
    echo "--- Set Demo ---"
    declare -A myset=()

    set_add()      { myset["$1"]=1; }
    set_remove()   { unset "myset[$1]"; }
    set_contains() { [[ -v myset[$1] ]]; }
    set_size()     { echo "${#myset[@]}"; }
    set_elements() { echo "${!myset[@]}"; }

    set_add "apple"
    set_add "banana"
    set_add "cherry"
    set_add "apple"  # duplicate
    echo "  Add: apple, banana, cherry, apple (dup)"
    echo "  Size: $(set_size)"
    echo "  Elements: $(set_elements)"
    set_contains "banana" && echo "  Contains banana: yes" || echo "  Contains banana: no"
    set_contains "grape" && echo "  Contains grape: yes" || echo "  Contains grape: no"
    set_remove "banana"
    echo "  After removing banana: $(set_elements)"
}

# === Exercise 4: Table Data Processor ===
# Problem: Load table data, add/remove columns, filter, and format output.
exercise_4() {
    echo "=== Exercise 4: Table Data Processor ==="

    # Simulated table using associative array
    declare -A table
    declare -a table_columns=("name" "department" "salary")
    local table_rows=0

    table_add_row() {
        table["${table_rows}_name"]="$1"
        table["${table_rows}_department"]="$2"
        table["${table_rows}_salary"]="$3"
        (( table_rows++ ))
    }

    table_print() {
        # Header
        printf "%-15s %-15s %10s\n" "${table_columns[@]}"
        printf "%-15s %-15s %10s\n" "---------------" "---------------" "----------"
        # Rows
        for (( i=0; i<table_rows; i++ )); do
            printf "%-15s %-15s %10s\n" \
                "${table["${i}_name"]}" \
                "${table["${i}_department"]}" \
                "${table["${i}_salary"]}"
        done
    }

    table_filter() {
        local col="$1"
        local op="$2"
        local val="$3"

        printf "%-15s %-15s %10s\n" "${table_columns[@]}"
        printf "%-15s %-15s %10s\n" "---------------" "---------------" "----------"

        for (( i=0; i<table_rows; i++ )); do
            local cell="${table["${i}_${col}"]}"
            local match=false

            case "$op" in
                "=")  [ "$cell" = "$val" ] && match=true ;;
                ">")  (( cell > val )) && match=true ;;
                "<")  (( cell < val )) && match=true ;;
                ">=") (( cell >= val )) && match=true ;;
            esac

            if $match; then
                printf "%-15s %-15s %10s\n" \
                    "${table["${i}_name"]}" \
                    "${table["${i}_department"]}" \
                    "${table["${i}_salary"]}"
            fi
        done
    }

    table_group_by() {
        local col="$1"
        declare -A groups
        declare -A group_count

        for (( i=0; i<table_rows; i++ )); do
            local key="${table["${i}_${col}"]}"
            local salary="${table["${i}_salary"]}"
            groups["$key"]=$(( ${groups[$key]:-0} + salary ))
            group_count["$key"]=$(( ${group_count[$key]:-0} + 1 ))
        done

        printf "%-15s %10s %10s\n" "$col" "count" "avg_salary"
        printf "%-15s %10s %10s\n" "---------------" "----------" "----------"
        for key in "${!groups[@]}"; do
            local avg=$(( groups[$key] / group_count[$key] ))
            printf "%-15s %10d %10d\n" "$key" "${group_count[$key]}" "$avg"
        done
    }

    # Populate data
    table_add_row "Alice"   "Engineering" 85000
    table_add_row "Bob"     "Marketing"   70000
    table_add_row "Charlie" "Engineering" 92000
    table_add_row "Diana"   "Sales"       68000
    table_add_row "Eve"     "Engineering" 88000
    table_add_row "Frank"   "Marketing"   72000

    echo "All data:"
    table_print
    echo ""

    echo "Filter: salary > 80000"
    table_filter "salary" ">" 80000
    echo ""

    echo "Group by department:"
    table_group_by "department"
}

# === Exercise 5: Log Aggregator ===
# Problem: Parse log files, extract structured data, generate statistics.
exercise_5() {
    echo "=== Exercise 5: Log Aggregator ==="

    # Create sample log file
    local log_file="/tmp/app_log_$$.log"
    cat > "$log_file" << 'EOF'
2024-02-13 10:00:01 [INFO] Application started
2024-02-13 10:00:05 [INFO] Database connected
2024-02-13 10:01:15 [WARN] Slow query detected (1.5s)
2024-02-13 10:02:30 [ERROR] Connection timeout to cache server
2024-02-13 10:03:00 [INFO] Retry succeeded for cache
2024-02-13 10:05:22 [ERROR] Failed to process request: invalid JSON
2024-02-13 10:06:00 [INFO] Request processed successfully
2024-02-13 10:07:45 [WARN] High memory usage: 85%
2024-02-13 10:08:10 [ERROR] Out of memory: allocation failed
2024-02-13 10:08:15 [INFO] Memory cleanup completed
2024-02-13 10:10:00 [INFO] Health check passed
EOF

    echo "--- Log File ---"
    cat "$log_file"
    echo ""

    # Count by level
    declare -A level_counts
    local total_lines=0

    while IFS= read -r line; do
        (( total_lines++ ))
        if [[ "$line" =~ \[([A-Z]+)\] ]]; then
            local level="${BASH_REMATCH[1]}"
            level_counts["$level"]=$(( ${level_counts[$level]:-0} + 1 ))
        fi
    done < "$log_file"

    echo "--- Statistics ---"
    echo "Total log lines: $total_lines"
    for level in INFO WARN ERROR; do
        printf "  %-8s: %d\n" "$level" "${level_counts[$level]:-0}"
    done

    # Calculate error rate
    local errors="${level_counts[ERROR]:-0}"
    if (( total_lines > 0 )); then
        local error_rate=$(( errors * 100 / total_lines ))
        echo "  Error rate: ${error_rate}%"
    fi

    echo ""
    echo "--- Error Messages ---"
    grep '\[ERROR\]' "$log_file" | while IFS= read -r line; do
        if [[ "$line" =~ \[ERROR\]\ (.+)$ ]]; then
            echo "  - ${BASH_REMATCH[1]}"
        fi
    done

    echo ""
    echo "--- Timeline (errors per minute) ---"
    grep '\[ERROR\]' "$log_file" | while IFS= read -r line; do
        if [[ "$line" =~ ^([0-9-]+\ [0-9]+:[0-9]+) ]]; then
            echo "${BASH_REMATCH[1]}"
        fi
    done | sort | uniq -c | while read -r count minute; do
        printf "  %s : %s error(s)\n" "$minute" "$count"
    done

    rm -f "$log_file"
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
