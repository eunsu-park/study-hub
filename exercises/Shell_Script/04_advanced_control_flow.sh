#!/bin/bash
# Exercises for Lesson 04: Advanced Control Flow
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Advanced Input Validator ===
# Problem: Validate different data types (email, URL, IP, phone, date)
# with detailed error messages and composite validation.
exercise_1() {
    echo "=== Exercise 1: Advanced Input Validator ==="

    validate() {
        local value="$1"
        local type="$2"
        shift 2
        local errors=()

        case "$type" in
            email)
                local pattern='^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if [[ ! "$value" =~ $pattern ]]; then
                    errors+=("Invalid email format")
                fi
                ;;
            ipv4)
                local pattern='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
                if [[ ! "$value" =~ $pattern ]]; then
                    errors+=("Invalid IPv4 format")
                else
                    IFS='.' read -ra octets <<< "$value"
                    for octet in "${octets[@]}"; do
                        if (( octet > 255 )); then
                            errors+=("Octet $octet exceeds 255")
                        fi
                    done
                fi
                ;;
            date)
                local pattern='^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                if [[ ! "$value" =~ $pattern ]]; then
                    errors+=("Invalid date format (expected YYYY-MM-DD)")
                else
                    local month="${value:5:2}"
                    local day="${value:8:2}"
                    if (( month < 1 || month > 12 )); then
                        errors+=("Month out of range: $month")
                    fi
                    if (( day < 1 || day > 31 )); then
                        errors+=("Day out of range: $day")
                    fi
                fi
                ;;
            phone)
                # Accept formats like (555) 123-4567 or 555-123-4567
                local digits
                digits=$(echo "$value" | tr -cd '0-9')
                if (( ${#digits} != 10 && ${#digits} != 11 )); then
                    errors+=("Phone number must have 10 or 11 digits (got ${#digits})")
                fi
                ;;
            password)
                local min_length=8
                # Parse optional arguments
                while [ $# -gt 0 ]; do
                    case "$1" in
                        --min-length) min_length="$2"; shift 2 ;;
                        --require-digit)
                            [[ ! "$value" =~ [0-9] ]] && errors+=("Must contain a digit")
                            shift ;;
                        --require-special)
                            [[ ! "$value" =~ [^a-zA-Z0-9] ]] && errors+=("Must contain a special character")
                            shift ;;
                        *) shift ;;
                    esac
                done
                if (( ${#value} < min_length )); then
                    errors+=("Too short: ${#value} < $min_length characters")
                fi
                ;;
            *)
                errors+=("Unknown validation type: $type")
                ;;
        esac

        if (( ${#errors[@]} == 0 )); then
            echo "  PASS: '$value' is a valid $type"
            return 0
        else
            echo "  FAIL: '$value' ($type)"
            for err in "${errors[@]}"; do
                echo "    - $err"
            done
            return 1
        fi
    }

    validate "user@example.com" email
    validate "invalid-email" email
    validate "192.168.1.1" ipv4
    validate "256.1.1.1" ipv4
    validate "2024-02-13" date
    validate "2024-13-40" date
    validate "(555) 123-4567" phone
    validate "12345" phone
    validate "MyP@ss1" password --min-length 8 --require-digit --require-special
    validate "MyP@ssw0rd!" password --min-length 8 --require-digit --require-special
}

# === Exercise 2: Expression Evaluator ===
# Problem: Parse and evaluate arithmetic expressions with variables
# and floating-point support via bc.
exercise_2() {
    echo "=== Exercise 2: Expression Evaluator ==="

    declare -A eval_vars

    eval_expr() {
        local expr="$1"
        shift

        # Load variables from arguments (x=5 format)
        for arg in "$@"; do
            if [[ "$arg" == *=* ]]; then
                local key="${arg%%=*}"
                local val="${arg#*=}"
                eval_vars["$key"]="$val"
            fi
        done

        # Handle variable assignment in expression
        if [[ "$expr" =~ ^([a-zA-Z_]+)\ *=\ *(.+)$ ]]; then
            local var_name="${BASH_REMATCH[1]}"
            local var_expr="${BASH_REMATCH[2]}"
            local result
            result=$(eval_expr "$var_expr")
            eval_vars["$var_name"]="$result"
            echo "$result"
            return
        fi

        # Substitute variables in the expression
        local processed="$expr"
        for key in "${!eval_vars[@]}"; do
            processed="${processed//$key/${eval_vars[$key]}}"
        done

        # Handle math functions
        processed="${processed//sqrt(/sqrt(}"
        processed="${processed//pow(/}"

        # Evaluate with bc for floating-point support
        local result
        result=$(echo "scale=4; $processed" | bc -l 2>/dev/null)

        if [ -z "$result" ]; then
            echo "Error: Invalid expression '$expr'" >&2
            return 1
        fi

        # Remove trailing zeros after decimal
        result=$(echo "$result" | sed 's/\.0*$//' | sed 's/\(\.[0-9]*[1-9]\)0*$/\1/')
        echo "$result"
    }

    echo "  2 + 3 * 4 = $(eval_expr '2 + 3 * 4')"
    echo "  10 / 3    = $(eval_expr '10 / 3')"
    echo "  2 ^ 10    = $(eval_expr '2 ^ 10')"

    eval_vars=()
    eval_expr "x = 5" > /dev/null
    eval_expr "y = 10" > /dev/null
    echo "  x=5, y=10, x+y = $(eval_expr 'x + y')"
    echo "  sqrt(16)  = $(echo "scale=4; sqrt(16)" | bc -l | sed 's/\.0*$//')"
}

# === Exercise 3: Pattern Matcher ===
# Problem: Match files against glob, regex, and extended glob patterns.
exercise_3() {
    echo "=== Exercise 3: Pattern Matcher ==="

    pattern_match() {
        local string="$1"
        shift

        local mode="" pattern="" exclude=""

        while [ $# -gt 0 ]; do
            case "$1" in
                --glob)    mode="glob"; pattern="$2"; shift 2 ;;
                --regex)   mode="regex"; pattern="$2"; shift 2 ;;
                --exclude) exclude="$2"; shift 2 ;;
                *) shift ;;
            esac
        done

        # Check exclusion first
        if [ -n "$exclude" ]; then
            if [[ "$string" == $exclude ]]; then
                echo "  '$string' excluded by pattern '$exclude'"
                return 1
            fi
        fi

        case "$mode" in
            glob)
                if [[ "$string" == $pattern ]]; then
                    echo "  '$string' matches glob '$pattern': YES"
                    return 0
                else
                    echo "  '$string' matches glob '$pattern': NO"
                    return 1
                fi
                ;;
            regex)
                if [[ "$string" =~ $pattern ]]; then
                    echo "  '$string' matches regex '$pattern': YES"
                    if [ ${#BASH_REMATCH[@]} -gt 1 ]; then
                        echo "    Groups: ${BASH_REMATCH[*]:1}"
                    fi
                    return 0
                else
                    echo "  '$string' matches regex '$pattern': NO"
                    return 1
                fi
                ;;
            *)
                echo "  Error: Specify --glob or --regex"
                return 2
                ;;
        esac
    }

    pattern_explain() {
        local pattern="$1"
        echo "  Pattern: $pattern"
        echo "  Explanation:"

        # Simple pattern explanation
        [[ "$pattern" == *'*'* ]] && echo "    * : matches any characters"
        [[ "$pattern" == *'?'* ]] && echo "    ? : matches exactly one character"
        [[ "$pattern" =~ \[.*\] ]] && echo "    [...] : matches character class"
        [[ "$pattern" == *'^'* ]] && echo "    ^ : anchor at start"
        [[ "$pattern" == *'$'* ]] && echo "    \$ : anchor at end"
        [[ "$pattern" == *'+'* ]] && echo "    + : one or more of preceding"
        [[ "$pattern" =~ \{[0-9] ]] && echo "    {n,m} : bounded repetition"
    }

    pattern_match "file.txt" --glob "*.txt"
    pattern_match "file.log" --glob "*.txt"
    pattern_match "temp_file.txt" --glob "*.txt" --exclude "temp_*"
    pattern_match "192.168.1.1" --regex '^([0-9]{1,3}\.){3}[0-9]{1,3}$'
    pattern_match "not_an_ip" --regex '^([0-9]{1,3}\.){3}[0-9]{1,3}$'
    echo ""
    pattern_explain '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
}

# === Exercise 4: Interactive Wizard ===
# Problem: Build multi-step forms with validation, conditionals, and summary.
# NOTE: This exercise simulates user input since it cannot be run interactively
# in an automated context.
exercise_4() {
    echo "=== Exercise 4: Interactive Wizard (simulated) ==="

    # Simulated wizard with preset answers
    local step=0
    declare -A wizard_data

    wizard_step() {
        local name="$1"
        local prompt="$2"
        local validator="$3"
        local default="$4"
        local value="$5"  # Pre-set value for simulation

        (( step++ ))
        echo "  Step $step: $prompt"

        # Use provided value (simulation mode)
        if [ -z "$value" ] && [ -n "$default" ]; then
            value="$default"
        fi

        echo "    -> $value"
        wizard_data["$name"]="$value"
    }

    wizard_summary() {
        echo ""
        echo "  --- Configuration Summary ---"
        for key in "${!wizard_data[@]}"; do
            printf "    %-15s : %s\n" "$key" "${wizard_data[$key]}"
        done
    }

    echo "  Setup Database Connection"
    echo "  ========================="
    wizard_step "host"     "Database host"  "" "localhost"    "db.example.com"
    wizard_step "port"     "Port number"    "" "5432"         "5432"
    wizard_step "dbname"   "Database name"  "" "myapp"        "production_db"
    wizard_step "ssl"      "Use SSL? (y/n)" "" "y"            "y"
    wizard_step "cert"     "Certificate path (since SSL=yes)" "" "" "/etc/ssl/cert.pem"

    wizard_summary

    echo ""
    echo "  [Simulated] Proceed with configuration? -> yes"
    echo "  Configuration applied successfully!"
}

# === Exercise 5: Smart Retry System ===
# Problem: Implement retry with multiple strategies (constant, linear,
# exponential) and jitter.
exercise_5() {
    echo "=== Exercise 5: Smart Retry System ==="

    retry() {
        local strategy="exponential"
        local max_attempts=5
        local initial_delay=1
        local max_delay=30
        local jitter=0
        local cmd=()

        # Parse arguments
        while [ $# -gt 0 ]; do
            case "$1" in
                --strategy)     strategy="$2"; shift 2 ;;
                --max-attempts) max_attempts="$2"; shift 2 ;;
                --initial-delay) initial_delay="$2"; shift 2 ;;
                --max-delay)    max_delay="$2"; shift 2 ;;
                --jitter)       jitter="$2"; shift 2 ;;
                --)             shift; cmd=("$@"); break ;;
                *)              cmd=("$@"); break ;;
            esac
        done

        local attempt=0
        local delay="$initial_delay"
        local total_start=$SECONDS

        while (( attempt < max_attempts )); do
            (( attempt++ ))
            echo "  Attempt $attempt/$max_attempts (delay: ${delay}s, strategy: $strategy)"

            # Execute command
            if "${cmd[@]}" 2>/dev/null; then
                local elapsed=$(( SECONDS - total_start ))
                echo "  SUCCESS after $attempt attempt(s), ${elapsed}s elapsed"
                return 0
            fi

            if (( attempt < max_attempts )); then
                echo "    Failed, waiting ${delay}s..."
                sleep "$delay" 2>/dev/null || true

                # Calculate next delay based on strategy
                case "$strategy" in
                    constant)
                        delay="$initial_delay"
                        ;;
                    linear)
                        delay=$(( initial_delay * (attempt + 1) ))
                        ;;
                    exponential)
                        delay=$(( initial_delay * (2 ** attempt) ))
                        ;;
                    fibonacci)
                        local a=1 b=1
                        for (( i=0; i<attempt; i++ )); do
                            local temp=$b
                            b=$(( a + b ))
                            a=$temp
                        done
                        delay=$a
                        ;;
                esac

                # Cap at max_delay
                (( delay > max_delay )) && delay=$max_delay
            fi
        done

        echo "  FAILED after $max_attempts attempts"
        return 1
    }

    # Simulate an unreliable command using a counter file
    local counter_file="/tmp/retry_counter_$$"
    echo "0" > "$counter_file"

    unreliable_cmd() {
        local count
        count=$(cat "$counter_file")
        (( count++ ))
        echo "$count" > "$counter_file"
        # Succeed on the 3rd attempt
        if (( count >= 3 )); then
            return 0
        fi
        return 1
    }

    echo "Test: Exponential backoff (succeeds on attempt 3)"
    retry --strategy exponential --max-attempts 5 --initial-delay 0 -- unreliable_cmd
    echo ""

    # Reset counter
    echo "0" > "$counter_file"

    echo "Test: Constant delay (succeeds on attempt 3)"
    retry --strategy constant --max-attempts 5 --initial-delay 0 -- unreliable_cmd

    rm -f "$counter_file"
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
