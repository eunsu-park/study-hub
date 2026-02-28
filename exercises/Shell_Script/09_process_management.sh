#!/bin/bash
# Exercises for Lesson 09: Process Management and Job Control
# Topic: Shell_Script
# Solutions to practice problems from the lesson.

# === Exercise 1: Parallel File Processor ===
# Problem: Process multiple files in parallel with configurable max
# concurrent jobs, track progress and handle errors.
exercise_1() {
    echo "=== Exercise 1: Parallel File Processor ==="

    local work_dir="/tmp/parallel_proc_$$"
    mkdir -p "$work_dir"

    # Create test files
    for i in {1..8}; do
        echo "content of file $i" > "$work_dir/file_${i}.txt"
    done

    local max_jobs=3
    local total=0
    local completed=0
    local failed=0
    local start_time=$SECONDS
    declare -A job_pids

    # Process a single file (simulate work)
    process_file() {
        local file="$1"
        local name
        name=$(basename "$file")
        echo "  [START] Processing $name (PID $$)"
        # Simulate variable work duration
        sleep "0.$((RANDOM % 5))"
        echo "  [DONE]  Processed $name"
    }

    # Count .txt files
    local files=("$work_dir"/*.txt)
    total=${#files[@]}

    echo "Processing $total files with max $max_jobs concurrent jobs..."
    echo ""

    local running=0

    for file in "${files[@]}"; do
        # Wait if we've reached max concurrent jobs
        while (( running >= max_jobs )); do
            # Wait for any child to finish
            wait -n 2>/dev/null || true
            (( running-- ))
            (( completed++ ))
        done

        # Launch in background
        process_file "$file" &
        job_pids[$!]="$file"
        (( running++ ))
    done

    # Wait for remaining jobs
    wait

    local elapsed=$(( SECONDS - start_time ))
    echo ""
    echo "--- Summary ---"
    echo "  Total files:    $total"
    echo "  Completed:      $total"
    echo "  Max concurrency: $max_jobs"
    echo "  Total time:     ${elapsed}s"

    rm -rf "$work_dir"
}

# === Exercise 2: Signal-Safe Download Manager ===
# Problem: Download multiple URLs in parallel, save progress to state file,
# resume on interrupt, clean up on SIGTERM.
exercise_2() {
    echo "=== Exercise 2: Signal-Safe Download Manager ==="

    local work_dir="/tmp/dlmgr_$$"
    local state_file="$work_dir/state.txt"
    mkdir -p "$work_dir/downloads"

    # Simulate URLs to download
    local urls=(
        "https://example.com/file1.tar.gz"
        "https://example.com/file2.tar.gz"
        "https://example.com/file3.tar.gz"
        "https://example.com/file4.tar.gz"
    )

    # Initialize state file
    for url in "${urls[@]}"; do
        echo "pending $url" >> "$state_file"
    done

    # Cleanup handler
    cleanup() {
        echo "  [CLEANUP] Saving state and cleaning up..."
        # In real usage: remove partial downloads
        echo "  [CLEANUP] State saved to $state_file"
    }

    # Install trap (local to this exercise, not global)
    # We simulate without actually trapping since this runs in a function

    # Simulate download
    simulate_download() {
        local url="$1"
        local filename
        filename=$(basename "$url")
        echo "  [DOWNLOAD] Starting: $filename"
        sleep "0.$((RANDOM % 3))"
        echo "  [DOWNLOAD] Completed: $filename"
        return 0
    }

    # Process downloads from state file
    echo "--- Processing Downloads ---"
    local line_num=0
    while IFS=' ' read -r status url; do
        (( line_num++ ))
        if [ "$status" = "pending" ]; then
            if simulate_download "$url"; then
                # Update state to completed
                sed -i.bak "${line_num}s/pending/completed/" "$state_file" 2>/dev/null || \
                    sed -i '' "${line_num}s/pending/completed/" "$state_file" 2>/dev/null || true
            else
                sed -i.bak "${line_num}s/pending/failed/" "$state_file" 2>/dev/null || \
                    sed -i '' "${line_num}s/pending/failed/" "$state_file" 2>/dev/null || true
            fi
        else
            echo "  [SKIP] $url (already $status)"
        fi
    done < "$state_file"

    echo ""
    echo "--- Final State ---"
    cat "$state_file" | sed 's/^/  /'

    cleanup
    rm -rf "$work_dir"
}

# === Exercise 3: Background Job Monitor ===
# Problem: Start multiple background jobs, monitor status, report completion,
# limit concurrent jobs, and retry failures.
exercise_3() {
    echo "=== Exercise 3: Background Job Monitor ==="

    local max_concurrent=3
    declare -A job_status
    declare -A job_names
    declare -a job_queue

    # Simulate a job
    run_job() {
        local name="$1"
        local duration="$2"
        local should_fail="${3:-false}"
        sleep "$duration"
        if [ "$should_fail" = "true" ]; then
            return 1
        fi
        return 0
    }

    # Submit jobs
    local jobs=(
        "backup:1:false"
        "compile:1:false"
        "test:1:true"      # This one will fail
        "deploy:1:false"
        "notify:1:false"
    )

    echo "--- Starting Job Monitor (max concurrent: $max_concurrent) ---"

    local running_pids=()
    local completed=0
    local failed=0
    local retried=0
    local total=${#jobs[@]}

    for job_spec in "${jobs[@]}"; do
        IFS=':' read -r name duration fail <<< "$job_spec"

        # Wait if at max concurrent
        while (( ${#running_pids[@]} >= max_concurrent )); do
            local new_pids=()
            for pid in "${running_pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                else
                    wait "$pid" 2>/dev/null
                    local exit_code=$?
                    local jname="${job_names[$pid]}"
                    if (( exit_code == 0 )); then
                        echo "  [OK]   $jname completed"
                        (( completed++ ))
                    else
                        echo "  [FAIL] $jname failed (exit: $exit_code)"
                        # Retry once
                        if [ "${job_status[$jname]}" != "retried" ]; then
                            echo "  [RETRY] Retrying $jname..."
                            run_job "$jname" "0.5" "false" &
                            local retry_pid=$!
                            job_names[$retry_pid]="$jname"
                            job_status["$jname"]="retried"
                            new_pids+=("$retry_pid")
                            (( retried++ ))
                        else
                            (( failed++ ))
                        fi
                    fi
                fi
            done
            running_pids=("${new_pids[@]}")
            sleep 0.2
        done

        # Start job
        echo "  [START] $name"
        run_job "$name" "$duration" "$fail" &
        local pid=$!
        running_pids+=("$pid")
        job_names[$pid]="$name"
    done

    # Wait for remaining jobs
    for pid in "${running_pids[@]}"; do
        wait "$pid" 2>/dev/null
        local exit_code=$?
        local jname="${job_names[$pid]}"
        if (( exit_code == 0 )); then
            echo "  [OK]   $jname completed"
            (( completed++ ))
        else
            if [ "${job_status[$jname]}" != "retried" ]; then
                echo "  [FAIL] $jname failed, retrying..."
                run_job "$jname" "0.5" "false" &
                wait $!
                echo "  [OK]   $jname retry completed"
                (( retried++ ))
            fi
            (( failed++ ))
        fi
    done

    echo ""
    echo "--- Summary ---"
    echo "  Total jobs: $total"
    echo "  Completed:  $completed"
    echo "  Retried:    $retried"
    echo "  Failed:     $failed"
}

# === Exercise 4: Coprocess Calculator Service ===
# Problem: Start a bc coprocess, provide CLI for expressions,
# support history, handle errors, and implement clean shutdown.
exercise_4() {
    echo "=== Exercise 4: Coprocess Calculator Service ==="

    # Start bc coprocess
    coproc CALC { bc -l 2>/dev/null; }

    declare -a calc_history

    # Calculate an expression via the coprocess
    calc_eval() {
        local expr="$1"

        # Send expression to bc
        echo "$expr" >&"${CALC[1]}"

        # Read result with timeout
        local result
        if read -t 2 -u "${CALC[0]}" result; then
            # Remove trailing zeros for cleaner output
            result=$(echo "$result" | sed 's/\.0*$//' | sed 's/\(\.[0-9]*[1-9]\)0*$/\1/')
            calc_history+=("$expr = $result")
            echo "$result"
            return 0
        else
            echo "Error: timeout or invalid expression"
            return 1
        fi
    }

    show_history() {
        echo "  --- Calculation History ---"
        if (( ${#calc_history[@]} == 0 )); then
            echo "    (empty)"
        else
            local start=0
            (( ${#calc_history[@]} > 10 )) && start=$(( ${#calc_history[@]} - 10 ))
            for (( i=start; i<${#calc_history[@]}; i++ )); do
                echo "    $((i+1)). ${calc_history[$i]}"
            done
        fi
    }

    # Simulated interactive session
    echo "--- Calculator Session ---"
    local expressions=(
        "2 + 3"
        "10 / 3"
        "2 ^ 10"
        "sqrt(144)"
        "4 * a(1)"
    )

    for expr in "${expressions[@]}"; do
        local result
        result=$(calc_eval "$expr")
        printf "  %-15s = %s\n" "$expr" "$result"
    done

    echo ""
    show_history

    # Shutdown coprocess
    echo ""
    echo "  Shutting down calculator..."
    eval "exec ${CALC[1]}>&-" 2>/dev/null
    wait "$CALC_PID" 2>/dev/null
    echo "  Calculator service stopped."
}

# === Exercise 5: Priority-Based Task Scheduler ===
# Problem: Accept tasks with priority levels, run with appropriate nice values,
# report status, limit concurrency, and implement cleanup.
exercise_5() {
    echo "=== Exercise 5: Priority-Based Task Scheduler ==="

    local max_tasks=5
    declare -A task_info
    declare -a active_pids

    # Submit a task with priority
    submit_task() {
        local name="$1"
        local priority="$2"
        local command="$3"

        local nice_value=0
        case "$priority" in
            high)   nice_value=0 ;;
            medium) nice_value=10 ;;
            low)    nice_value=19 ;;
        esac

        # Check concurrent limit
        if (( ${#active_pids[@]} >= max_tasks )); then
            echo "  [QUEUE] $name - waiting (max $max_tasks concurrent)"
            # Wait for any to finish
            wait -n 2>/dev/null || true
            # Remove finished PIDs
            local new_pids=()
            for pid in "${active_pids[@]}"; do
                kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
            done
            active_pids=("${new_pids[@]}")
        fi

        # Start task with nice (simulate with sleep since nice may need permissions)
        echo "  [START] $name (priority=$priority, nice=$nice_value)"
        (
            # In a real system: nice -n $nice_value $command
            eval "$command"
        ) &
        local pid=$!
        active_pids+=("$pid")
        task_info["$pid"]="$name|$priority|running"
    }

    # Show status of all tasks
    show_status() {
        echo "  --- Task Status ---"
        printf "  %-5s %-15s %-8s %s\n" "PID" "Name" "Priority" "Status"
        printf "  %-5s %-15s %-8s %s\n" "-----" "---------------" "--------" "-------"
        for pid in "${!task_info[@]}"; do
            IFS='|' read -r name priority status <<< "${task_info[$pid]}"
            if kill -0 "$pid" 2>/dev/null; then
                status="running"
            else
                wait "$pid" 2>/dev/null
                if (( $? == 0 )); then
                    status="completed"
                else
                    status="failed"
                fi
            fi
            printf "  %-5s %-15s %-8s %s\n" "$pid" "$name" "$priority" "$status"
        done
    }

    # Submit tasks with different priorities
    echo "--- Submitting Tasks ---"
    submit_task "critical_backup" "high"   "sleep 0.5"
    submit_task "build_project"   "high"   "sleep 0.5"
    submit_task "run_tests"       "medium" "sleep 0.5"
    submit_task "gen_docs"        "medium" "sleep 0.5"
    submit_task "cleanup_logs"    "low"    "sleep 0.5"

    echo ""
    echo "--- Status (during execution) ---"
    show_status

    # Wait for all
    wait

    echo ""
    echo "--- Final Status ---"
    show_status

    # Cleanup
    echo ""
    echo "  All tasks completed. Cleanup done."
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
