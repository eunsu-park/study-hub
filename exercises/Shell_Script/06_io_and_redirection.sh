#!/bin/bash
# Exercises for Lesson 06: I/O and Redirection
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Multi-Target Logger ===
# Problem: Logging system that writes to multiple files by level,
# adds timestamps and hostname, and displays on appropriate streams.
exercise_1() {
    echo "=== Exercise 1: Multi-Target Logger ==="

    local log_dir="/tmp/logger_test_$$"
    mkdir -p "$log_dir"

    local all_log="$log_dir/all.log"
    local error_log="$log_dir/error.log"
    local important_log="$log_dir/important.log"

    log_message() {
        local level="$1"
        shift
        local message="$*"
        local timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local hostname
        hostname=$(hostname -s 2>/dev/null || echo "localhost")

        local entry="[$timestamp] [$hostname] [$level] $message"

        # Always write to all.log
        echo "$entry" >> "$all_log"

        # Write ERROR to error.log
        if [ "$level" = "ERROR" ]; then
            echo "$entry" >> "$error_log"
        fi

        # Write WARN and ERROR to important.log
        if [ "$level" = "ERROR" ] || [ "$level" = "WARN" ]; then
            echo "$entry" >> "$important_log"
        fi

        # Display: ERROR/WARN on stderr, others on stdout
        if [ "$level" = "ERROR" ] || [ "$level" = "WARN" ]; then
            echo "$entry" >&2
        else
            echo "$entry"
        fi
    }

    # Test logging
    log_message "INFO"  "Application started"
    log_message "DEBUG" "Loading configuration"
    log_message "INFO"  "Database connected"
    log_message "WARN"  "Cache miss rate high: 45%"
    log_message "ERROR" "Failed to connect to external API"
    log_message "INFO"  "Retrying connection..."
    log_message "ERROR" "Retry failed after 3 attempts"

    echo ""
    echo "--- all.log ($(wc -l < "$all_log") lines) ---"
    cat "$all_log"
    echo ""
    echo "--- error.log ($(wc -l < "$error_log") lines) ---"
    cat "$error_log"
    echo ""
    echo "--- important.log ($(wc -l < "$important_log") lines) ---"
    cat "$important_log"

    rm -rf "$log_dir"
}

# === Exercise 2: Pipeline Monitor ===
# Problem: Run a multi-stage pipeline, monitor PIPESTATUS, log progress,
# and report which stage failed.
exercise_2() {
    echo "=== Exercise 2: Pipeline Monitor ==="

    run_monitored_pipeline() {
        local start_time=$SECONDS
        local log_file="/tmp/pipeline_$$.log"

        echo "Starting pipeline..." | tee "$log_file"

        # Stage functions
        stage_generate() {
            echo "  [Stage 1] Generating data..."
            for i in $(seq 1 10); do
                echo "line $i: data_$(( RANDOM % 100 ))"
            done
        }

        stage_filter() {
            echo "  [Stage 2] Filtering..." >&2
            grep "data_[0-9]"  # Keep all lines with data_
        }

        stage_transform() {
            echo "  [Stage 3] Transforming..." >&2
            while IFS= read -r line; do
                echo "${line^^}"  # Uppercase
            done
        }

        stage_count() {
            echo "  [Stage 4] Counting..." >&2
            wc -l
        }

        # Run pipeline
        local result
        result=$(stage_generate | stage_filter | stage_transform | stage_count)
        local pipeline_status=("${PIPESTATUS[@]}")

        # Report results
        echo ""
        echo "Pipeline results:"
        local stage_names=("Generate" "Filter" "Transform" "Count")
        local all_ok=true

        for i in "${!pipeline_status[@]}"; do
            local status="${pipeline_status[$i]}"
            local name="${stage_names[$i]:-Stage$i}"
            if (( status == 0 )); then
                echo "  [OK]   $name (exit: $status)"
            else
                echo "  [FAIL] $name (exit: $status)"
                all_ok=false
            fi
        done

        echo ""
        if $all_ok; then
            echo "Pipeline completed successfully!"
            echo "Result: $result lines processed"
        else
            echo "Pipeline had failures!"
        fi

        local elapsed=$(( SECONDS - start_time ))
        echo "Total time: ${elapsed}s"

        rm -f "$log_file"
    }

    run_monitored_pipeline
}

# === Exercise 3: FIFO-Based Queue System ===
# Problem: Implement a simple job queue using named pipes with submit,
# process, and status tracking.
exercise_3() {
    echo "=== Exercise 3: FIFO-Based Queue System ==="

    local queue_dir="/tmp/jobqueue_$$"
    mkdir -p "$queue_dir"

    declare -A job_status

    # Submit a job (add to queue)
    job_submit() {
        local job_id="job_$(date +%s%N | tail -c 6)"
        local command="$*"

        job_status["$job_id"]="pending"
        echo "  Submitted: $job_id -> $command"
        echo "$job_id"
    }

    # Process a job
    job_process() {
        local job_id="$1"
        local command="$2"

        job_status["$job_id"]="running"
        echo "  Processing: $job_id"

        # Execute the command (simulated)
        if eval "$command" > /dev/null 2>&1; then
            job_status["$job_id"]="completed"
            echo "  Completed: $job_id"
        else
            job_status["$job_id"]="failed"
            echo "  Failed: $job_id"
        fi
    }

    # Show status of all jobs
    job_show_status() {
        echo "  Job Status:"
        for jid in "${!job_status[@]}"; do
            printf "    %-15s : %s\n" "$jid" "${job_status[$jid]}"
        done
    }

    # Submit jobs
    id1=$(job_submit "echo hello")
    id2=$(job_submit "sleep 0")
    id3=$(job_submit "false")

    echo ""
    echo "After submission:"
    job_show_status

    echo ""
    echo "Processing jobs..."
    job_process "$id1" "echo hello"
    job_process "$id2" "sleep 0"
    job_process "$id3" "false"

    echo ""
    echo "After processing:"
    job_show_status

    rm -rf "$queue_dir"
}

# === Exercise 4: Configuration Validator ===
# Problem: Read config, validate syntax, atomically replace if valid,
# keep old config if invalid, maintain backups.
exercise_4() {
    echo "=== Exercise 4: Configuration Validator ==="

    local config_dir="/tmp/cfgval_$$"
    mkdir -p "$config_dir/backups"

    local config_file="$config_dir/app.conf"

    # Create initial config
    cat > "$config_file" << 'EOF'
host = localhost
port = 8080
debug = true
EOF

    validate_config() {
        local file="$1"
        local errors=0

        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*$ ]] && continue
            [[ "$line" =~ ^[[:space:]]*# ]] && continue

            if [[ ! "$line" =~ ^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_]*[[:space:]]*= ]]; then
                echo "  Syntax error: $line"
                (( errors++ ))
            fi
        done < "$file"

        return $errors
    }

    apply_config() {
        local new_content="$1"
        local target="$2"
        local backup_dir="$3"
        local dry_run="${4:-false}"

        # Write new content to temp file
        local tmpfile
        tmpfile=$(mktemp "${target}.XXXXXX")
        echo "$new_content" > "$tmpfile"

        # Validate
        if validate_config "$tmpfile"; then
            if $dry_run; then
                echo "  [DRY RUN] Would apply new configuration"
                rm -f "$tmpfile"
                return 0
            fi

            # Create backup
            if [ -f "$target" ]; then
                local backup="$backup_dir/$(basename "$target").$(date +%Y%m%d_%H%M%S)"
                cp "$target" "$backup"
                echo "  Backup created: $backup"

                # Keep only last 5 backups
                ls -t "$backup_dir"/*.conf.* 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
            fi

            # Atomic replace
            mv "$tmpfile" "$target"
            echo "  Configuration applied successfully"
            return 0
        else
            echo "  Configuration invalid, keeping existing config"
            rm -f "$tmpfile"
            return 1
        fi
    }

    echo "Current config:"
    cat "$config_file" | sed 's/^/  /'
    echo ""

    # Apply valid config
    echo "Applying valid config..."
    apply_config "host = production.example.com
port = 9090
workers = 4" "$config_file" "$config_dir/backups"

    echo ""
    echo "Updated config:"
    cat "$config_file" | sed 's/^/  /'
    echo ""

    # Try applying invalid config
    echo "Applying invalid config..."
    apply_config "this is not valid config
!!!" "$config_file" "$config_dir/backups"

    echo ""
    echo "Config unchanged after invalid attempt:"
    cat "$config_file" | sed 's/^/  /'
    echo ""

    # Dry run
    echo "Dry run test..."
    apply_config "host = test
port = 1234" "$config_file" "$config_dir/backups" true

    rm -rf "$config_dir"
}

# === Exercise 5: Stream Processor with FDs ===
# Problem: Open multiple input streams, merge with timestamps,
# filter and split output, maintain statistics.
exercise_5() {
    echo "=== Exercise 5: Stream Processor with FDs ==="

    local work_dir="/tmp/streamproc_$$"
    mkdir -p "$work_dir"

    # Create source files
    echo -e "INFO: server started\nERROR: disk full\nINFO: request handled" > "$work_dir/stream1.txt"
    echo -e "WARN: high memory\nINFO: cache hit\nERROR: timeout" > "$work_dir/stream2.txt"
    echo -e "INFO: user login\nINFO: user logout\nWARN: slow query" > "$work_dir/stream3.txt"

    local info_out="$work_dir/info.out"
    local warn_out="$work_dir/warn.out"
    local error_out="$work_dir/error.out"

    local lines_total=0
    local lines_info=0
    local lines_warn=0
    local lines_error=0

    # Process streams using file descriptors
    exec 3< "$work_dir/stream1.txt"
    exec 4< "$work_dir/stream2.txt"
    exec 5< "$work_dir/stream3.txt"

    process_line() {
        local stream_id="$1"
        local line="$2"
        local timestamp
        timestamp=$(date '+%H:%M:%S')

        local tagged="[$timestamp] [stream$stream_id] $line"
        (( lines_total++ ))

        case "$line" in
            INFO:*)
                echo "$tagged" >> "$info_out"
                (( lines_info++ ))
                ;;
            WARN:*)
                echo "$tagged" >> "$warn_out"
                (( lines_warn++ ))
                ;;
            ERROR:*)
                echo "$tagged" >> "$error_out"
                (( lines_error++ ))
                ;;
        esac
    }

    # Read from all three streams
    while IFS= read -u 3 -r line; do
        process_line 1 "$line"
    done

    while IFS= read -u 4 -r line; do
        process_line 2 "$line"
    done

    while IFS= read -u 5 -r line; do
        process_line 3 "$line"
    done

    # Close file descriptors
    exec 3<&- 4<&- 5<&-

    echo "--- Statistics ---"
    echo "  Total lines processed: $lines_total"
    echo "  INFO:  $lines_info"
    echo "  WARN:  $lines_warn"
    echo "  ERROR: $lines_error"

    echo ""
    echo "--- info.out ---"
    [ -f "$info_out" ] && cat "$info_out" | sed 's/^/  /'

    echo ""
    echo "--- warn.out ---"
    [ -f "$warn_out" ] && cat "$warn_out" | sed 's/^/  /'

    echo ""
    echo "--- error.out ---"
    [ -f "$error_out" ] && cat "$error_out" | sed 's/^/  /'

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
