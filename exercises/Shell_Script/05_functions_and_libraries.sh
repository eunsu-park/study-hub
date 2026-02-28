#!/bin/bash
# Exercises for Lesson 05: Functions and Libraries
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: String Utility Library ===
# Problem: Implement str_reverse, str_is_palindrome, str_count_words,
# str_capitalize, and str_remove_duplicates.
exercise_1() {
    echo "=== Exercise 1: String Utility Library ==="

    # Reverse a string
    str_reverse() {
        local str="$1"
        local reversed=""
        local i
        for (( i=${#str}-1; i>=0; i-- )); do
            reversed+="${str:$i:1}"
        done
        echo "$reversed"
    }

    # Check if string is palindrome (returns 0 for true)
    str_is_palindrome() {
        local str="${1,,}"  # Convert to lowercase
        str="${str// /}"    # Remove spaces
        local reversed
        reversed=$(str_reverse "$str")
        [ "$str" = "$reversed" ]
    }

    # Count words in a string
    str_count_words() {
        local str="$1"
        local count=0
        for word in $str; do
            (( count++ ))
        done
        echo "$count"
    }

    # Capitalize first letter of each word
    str_capitalize() {
        local str="$1"
        local result=""
        local word
        for word in $str; do
            # Capitalize first char, append rest
            result+="${word^} "
        done
        # Remove trailing space
        echo "${result% }"
    }

    # Remove consecutive duplicate characters
    str_remove_duplicates() {
        local str="$1"
        local result=""
        local prev=""
        local i char
        for (( i=0; i<${#str}; i++ )); do
            char="${str:$i:1}"
            if [ "$char" != "$prev" ]; then
                result+="$char"
            fi
            prev="$char"
        done
        echo "$result"
    }

    echo "str_reverse 'hello':       $(str_reverse 'hello')"
    echo "str_reverse 'bash':        $(str_reverse 'bash')"
    echo ""

    echo "str_is_palindrome 'racecar':"
    str_is_palindrome "racecar" && echo "  YES" || echo "  NO"
    echo "str_is_palindrome 'hello':"
    str_is_palindrome "hello" && echo "  YES" || echo "  NO"
    echo "str_is_palindrome 'A man a plan a canal Panama':"
    str_is_palindrome "amanaplanacanalpanama" && echo "  YES" || echo "  NO"
    echo ""

    echo "str_count_words 'hello world foo bar':  $(str_count_words 'hello world foo bar')"
    echo "str_count_words 'single':               $(str_count_words 'single')"
    echo ""

    echo "str_capitalize 'hello world':            $(str_capitalize 'hello world')"
    echo "str_capitalize 'the quick brown fox':    $(str_capitalize 'the quick brown fox')"
    echo ""

    echo "str_remove_duplicates 'booook':          $(str_remove_duplicates 'booook')"
    echo "str_remove_duplicates 'aabbccdd':        $(str_remove_duplicates 'aabbccdd')"
    echo "str_remove_duplicates 'hello':           $(str_remove_duplicates 'hello')"
}

# === Exercise 2: Recursive File Search ===
# Problem: Recursively search directories for files matching a pattern,
# returning results via nameref. Handle symlinks to avoid infinite loops.
exercise_2() {
    echo "=== Exercise 2: Recursive File Search ==="

    # Create test directory structure
    local test_dir="/tmp/find_test_$$"
    mkdir -p "$test_dir/sub1/sub2"
    mkdir -p "$test_dir/sub3"
    touch "$test_dir/file1.txt" "$test_dir/file2.log"
    touch "$test_dir/sub1/file3.txt" "$test_dir/sub1/file4.py"
    touch "$test_dir/sub1/sub2/file5.txt"
    touch "$test_dir/sub3/file6.txt" "$test_dir/sub3/file7.log"

    find_files() {
        local dir="$1"
        local pattern="$2"
        local -n results_ref=$3
        local -n stats_ref=$4
        local depth="${5:-0}"
        local -A visited_dirs

        # Track stats
        stats_ref[searched]="${stats_ref[searched]:-0}"
        stats_ref[matched]="${stats_ref[matched]:-0}"

        # Resolve the real path to avoid symlink loops
        local real_dir
        real_dir=$(cd "$dir" 2>/dev/null && pwd -P) || return

        # Skip already visited directories
        if [ -n "${visited_dirs[$real_dir]:-}" ]; then
            return
        fi
        visited_dirs["$real_dir"]=1

        local item
        for item in "$dir"/*; do
            [ -e "$item" ] || continue

            if [ -d "$item" ]; then
                # Recurse into subdirectories
                find_files "$item" "$pattern" "$3" "$4" $(( depth + 1 ))
            elif [ -f "$item" ]; then
                (( stats_ref[searched]++ ))
                local basename
                basename=$(basename "$item")
                if [[ "$basename" == $pattern ]]; then
                    results_ref+=("$item")
                    (( stats_ref[matched]++ ))
                fi
            fi
        done
    }

    declare -a found_files
    declare -A search_stats

    echo "Searching in $test_dir for *.txt ..."
    find_files "$test_dir" "*.txt" found_files search_stats

    echo "Files searched: ${search_stats[searched]}"
    echo "Files matched:  ${search_stats[matched]}"
    echo ""
    echo "Matching files:"
    for f in "${found_files[@]}"; do
        echo "  $f"
    done

    # Cleanup
    rm -rf "$test_dir"
}

# === Exercise 3: Calculator with Callbacks ===
# Problem: Build a calculator that uses registered callback functions for
# each operation and maintains history.
exercise_3() {
    echo "=== Exercise 3: Calculator with Callbacks ==="

    declare -A calc_ops
    declare -a calc_history

    # Register an operation
    register_op() {
        local symbol="$1"
        local func_name="$2"
        calc_ops["$symbol"]="$func_name"
    }

    # Built-in operation callbacks
    calc_add()      { echo $(( $1 + $2 )); }
    calc_subtract() { echo $(( $1 - $2 )); }
    calc_multiply() { echo $(( $1 * $2 )); }
    calc_divide() {
        if (( $2 == 0 )); then
            echo "Error: division by zero" >&2
            return 1
        fi
        echo $(( $1 / $2 ))
    }
    calc_modulo() { echo $(( $1 % $2 )); }
    calc_power() {
        local result=1
        local i
        for (( i=0; i<$2; i++ )); do
            (( result *= $1 ))
        done
        echo "$result"
    }

    # Register built-in operations
    register_op "+" "calc_add"
    register_op "-" "calc_subtract"
    register_op "*" "calc_multiply"
    register_op "/" "calc_divide"
    register_op "%" "calc_modulo"
    register_op "**" "calc_power"

    # Calculator function
    calc() {
        local a="$1"
        local op="$2"
        local b="$3"

        # Validate inputs
        if [[ ! "$a" =~ ^-?[0-9]+$ ]] || [[ ! "$b" =~ ^-?[0-9]+$ ]]; then
            echo "Error: operands must be integers" >&2
            return 1
        fi

        local func="${calc_ops[$op]:-}"
        if [ -z "$func" ]; then
            echo "Error: unknown operator '$op'" >&2
            echo "Available: ${!calc_ops[*]}" >&2
            return 1
        fi

        local result
        result=$($func "$a" "$b")
        local status=$?

        if (( status == 0 )); then
            calc_history+=("$a $op $b = $result")
            echo "$result"
        fi

        return $status
    }

    # Show history
    calc_show_history() {
        echo "  History (last ${#calc_history[@]} calculations):"
        for entry in "${calc_history[@]}"; do
            echo "    $entry"
        done
    }

    echo "Basic operations:"
    echo "  10 + 5  = $(calc 10 '+' 5)"
    echo "  10 - 3  = $(calc 10 '-' 3)"
    echo "  6 * 7   = $(calc 6 '*' 7)"
    echo "  20 / 4  = $(calc 20 '/' 4)"
    echo "  17 % 5  = $(calc 17 '%' 5)"
    echo "  2 ** 10 = $(calc 2 '**' 10)"

    echo ""
    echo "Custom operation (register 'max'):"
    calc_max() { (( $1 > $2 )) && echo "$1" || echo "$2"; }
    register_op "max" "calc_max"
    echo "  15 max 23 = $(calc 15 'max' 23)"

    echo ""
    calc_show_history
}

# === Exercise 4: Configuration Manager ===
# Problem: Build config management with load, get, set, save, and validation.
exercise_4() {
    echo "=== Exercise 4: Configuration Manager ==="

    declare -A cfg_data

    config::load() {
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
            cfg_data["$key"]="$value"
        done < "$file"
        echo "  Loaded $(wc -l < "$file") lines from $file"
    }

    config::get() {
        local key="$1"
        local default="${2:-}"
        echo "${cfg_data[$key]:-$default}"
    }

    config::set() {
        local key="$1"
        local value="$2"
        local validator="${3:-}"

        if [ -n "$validator" ]; then
            if ! $validator "$value"; then
                echo "  Validation failed for $key=$value" >&2
                return 1
            fi
        fi
        cfg_data["$key"]="$value"
    }

    config::save() {
        local file="$1"
        {
            echo "# Configuration saved on $(date)"
            for key in $(echo "${!cfg_data[@]}" | tr ' ' '\n' | sort); do
                echo "$key = ${cfg_data[$key]}"
            done
        } > "$file"
        echo "  Saved to $file"
    }

    config::list() {
        for key in $(echo "${!cfg_data[@]}" | tr ' ' '\n' | sort); do
            printf "    %-15s = %s\n" "$key" "${cfg_data[$key]}"
        done
    }

    # Validator callback
    validate_port() {
        [[ "$1" =~ ^[0-9]+$ ]] && (( $1 >= 1 && $1 <= 65535 ))
    }

    # Test
    local config_file="/tmp/cfg_test_$$.conf"
    cat > "$config_file" << 'EOF'
# Test config
host = localhost
port = 8080
debug = true
EOF

    config::load "$config_file"

    echo "  Current config:"
    config::list

    echo ""
    echo "  Get: host = $(config::get 'host')"
    echo "  Get: missing = $(config::get 'missing' 'fallback')"

    echo ""
    echo "  Set: port = 9090 (with validation)"
    config::set "port" "9090" validate_port
    echo "  port = $(config::get 'port')"

    echo ""
    echo "  Set: port = 99999 (should fail validation)"
    config::set "port" "99999" validate_port 2>&1

    echo ""
    local save_file="/tmp/cfg_saved_$$.conf"
    config::save "$save_file"
    echo "  Saved config:"
    cat "$save_file" | sed 's/^/    /'

    rm -f "$config_file" "$save_file"
}

# === Exercise 5: Function Performance Profiler ===
# Problem: Wrap functions to measure execution time, track call count,
# and generate performance reports.
exercise_5() {
    echo "=== Exercise 5: Function Performance Profiler ==="

    declare -A prof_calls
    declare -A prof_total_time
    declare -A prof_min_time
    declare -A prof_max_time

    profile() {
        local func_name="$1"
        shift

        # Record start time
        local start_ns
        start_ns=$(date +%s%N 2>/dev/null || echo "$(date +%s)000000000")

        # Execute the function
        "$func_name" "$@"
        local status=$?

        # Record end time
        local end_ns
        end_ns=$(date +%s%N 2>/dev/null || echo "$(date +%s)000000000")

        # Calculate duration in microseconds
        local duration_us=$(( (end_ns - start_ns) / 1000 ))

        # Update statistics
        prof_calls["$func_name"]=$(( ${prof_calls[$func_name]:-0} + 1 ))
        prof_total_time["$func_name"]=$(( ${prof_total_time[$func_name]:-0} + duration_us ))

        if [ -z "${prof_min_time[$func_name]:-}" ] || (( duration_us < prof_min_time[$func_name] )); then
            prof_min_time["$func_name"]=$duration_us
        fi
        if [ -z "${prof_max_time[$func_name]:-}" ] || (( duration_us > prof_max_time[$func_name] )); then
            prof_max_time["$func_name"]=$duration_us
        fi

        return $status
    }

    profile_report() {
        echo "  --- Performance Report ---"
        printf "  %-20s %8s %12s %12s %12s %12s\n" \
            "Function" "Calls" "Total(us)" "Avg(us)" "Min(us)" "Max(us)"
        printf "  %-20s %8s %12s %12s %12s %12s\n" \
            "--------------------" "--------" "------------" "------------" "------------" "------------"

        for func in "${!prof_calls[@]}"; do
            local calls="${prof_calls[$func]}"
            local total="${prof_total_time[$func]}"
            local avg=$(( total / calls ))
            local min="${prof_min_time[$func]}"
            local max="${prof_max_time[$func]}"

            printf "  %-20s %8d %12d %12d %12d %12d\n" \
                "$func" "$calls" "$total" "$avg" "$min" "$max"
        done
    }

    # Test functions
    fast_function() {
        local sum=0
        for (( i=0; i<100; i++ )); do
            (( sum += i ))
        done
    }

    medium_function() {
        local sum=0
        for (( i=0; i<1000; i++ )); do
            (( sum += i ))
        done
    }

    slow_function() {
        sleep 0.01
    }

    # Profile the functions
    echo "  Running profiled functions..."
    for i in {1..5}; do
        profile fast_function
        profile medium_function
        profile slow_function
    done

    echo ""
    profile_report
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
