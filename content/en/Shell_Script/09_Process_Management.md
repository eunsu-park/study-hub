# Lesson 09: Process Management and Job Control

**Difficulty**: ⭐⭐⭐

**Previous**: [Regular Expressions in Bash](./08_Regex_in_Bash.md) | **Next**: [Error Handling and Debugging](./10_Error_Handling.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how background processes work and retrieve their PIDs using `$!`
2. Apply the `wait` command to synchronize background jobs by PID or job number
3. Implement parallel execution patterns with configurable concurrency limits
4. Distinguish between subshells and command grouping and their variable scope implications
5. Identify common Unix signals and describe their default behavior
6. Write `trap` handlers to catch signals and perform cleanup on script exit
7. Build idempotent cleanup functions that safely remove temporary files, release locks, and terminate child processes
8. Use coprocesses (`coproc`) for bidirectional inter-process communication in Bash

---

Production scripts frequently need to run tasks concurrently, coordinate multiple background jobs, and shut down gracefully when interrupted. Whether you are processing files in parallel to cut a 10-minute job to 2 minutes, or ensuring a cron job cleans up its temp files even after a crash, mastering process management and signal handling is essential for writing scripts that behave reliably under real-world conditions.

## 1. Background Processes

Shell scripts can run processes in the background, enabling parallel execution and improved performance.

### Basic Background Execution

```bash
#!/bin/bash

# Run command in background with &
sleep 10 &
echo "Sleep started in background"

# The PID of the last background process
SLEEP_PID=$!
echo "Sleep PID: $SLEEP_PID"

# Continue with other work
echo "Doing other work..."

# Wait for background process to complete
wait $SLEEP_PID
echo "Sleep completed"
```

### The wait Command

```bash
#!/bin/bash

# Start multiple background processes
sleep 2 &
PID1=$!

sleep 3 &
PID2=$!

sleep 1 &
PID3=$!

echo "Started 3 background processes: $PID1, $PID2, $PID3"

# Wait for all background jobs
wait
echo "All processes completed"

# Wait for specific PID
sleep 5 &
SPECIFIC_PID=$!
wait $SPECIFIC_PID
echo "Specific process $SPECIFIC_PID completed with exit code: $?"

# Bash 4.3+: wait for any job to complete
if [ "${BASH_VERSINFO[0]}" -ge 4 ] && [ "${BASH_VERSINFO[1]}" -ge 3 ]; then
    sleep 2 &
    sleep 4 &
    sleep 1 &

    wait -n  # Wait for next job to complete
    echo "First job completed"

    wait -n
    echo "Second job completed"

    wait
    echo "All remaining jobs completed"
fi
```

### Job Control Commands

```bash
#!/bin/bash

# Start a long-running process
sleep 100 &

# List all jobs
jobs
# Output: [1]+ Running    sleep 100 &

# List with PIDs
jobs -l
# Output: [1]+ 12345 Running    sleep 100 &

# List only running jobs
jobs -r

# List only stopped jobs
jobs -s

# Bring job to foreground
# fg %1    # (uncomment in interactive shell)

# Send job to background (if stopped)
# bg %1    # (uncomment in interactive shell)

# Reference jobs:
# %1       - Job number 1
# %?sleep  - Job whose command contains "sleep"
# %%       - Current job
# %+       - Current job (same as %%)
# %-       - Previous job

# Kill a background job
kill %1
```

### Job Control Example

```bash
#!/bin/bash

# Function to demonstrate job control
job_control_demo() {
    echo "Starting 3 jobs..."

    (sleep 5; echo "Job 1 done") &
    (sleep 3; echo "Job 2 done") &
    (sleep 7; echo "Job 3 done") &

    # Show all jobs
    jobs

    # Wait for job 2 specifically
    wait %2
    echo "Job 2 has completed"

    # Wait for all remaining
    wait
    echo "All jobs completed"
}

job_control_demo
```

## 2. Parallel Execution

Running tasks in parallel can dramatically speed up scripts that perform independent operations.

### Basic Parallel Pattern

```bash
#!/bin/bash

# Sequential execution (slow)
sequential() {
    for i in {1..5}; do
        sleep 1
        echo "Task $i completed"
    done
}

# Parallel execution (fast)
parallel_basic() {
    for i in {1..5}; do
        (
            sleep 1
            echo "Task $i completed"
        ) &
    done
    wait
}

echo "Sequential (5 seconds):"
time sequential

echo -e "\nParallel (1 second):"
time parallel_basic
```

### Tracking Background PIDs

```bash
#!/bin/bash

# Store PIDs in an array
pids=()

for i in {1..5}; do
    sleep $((RANDOM % 3 + 1)) &
    pids+=($!)
    echo "Started job $i with PID ${pids[-1]}"
done

# Wait for each PID and check exit status
for pid in "${pids[@]}"; do
    wait "$pid"
    status=$?
    echo "PID $pid exited with status $status"
done
```

### Limiting Concurrency

```bash
#!/bin/bash

# Run at most N jobs in parallel
MAX_JOBS=3

run_with_limit() {
    local max_jobs=$1
    shift
    local jobs=("$@")

    local pids=()
    local count=0

    for job in "${jobs[@]}"; do
        # If at limit, wait for one to complete
        if [ "${#pids[@]}" -ge "$max_jobs" ]; then
            wait -n  # Bash 4.3+
            # Remove completed PIDs
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    unset 'pids[$i]'
                fi
            done
            pids=("${pids[@]}")  # Re-index array
        fi

        # Start new job
        eval "$job" &
        pids+=($!)
        echo "Started job: $job (PID: $!)"
    done

    # Wait for remaining jobs
    wait
    echo "All jobs completed"
}

# Example usage
jobs=(
    "sleep 2; echo 'Job 1 done'"
    "sleep 1; echo 'Job 2 done'"
    "sleep 3; echo 'Job 3 done'"
    "sleep 1; echo 'Job 4 done'"
    "sleep 2; echo 'Job 5 done'"
)

run_with_limit $MAX_JOBS "${jobs[@]}"
```

### Using xargs for Parallel Execution

```bash
#!/bin/bash

# Process files in parallel with xargs
process_file() {
    local file=$1
    echo "Processing $file..."
    sleep 1
    echo "$file processed"
}

export -f process_file

# Run 4 jobs in parallel
echo -e "file1.txt\nfile2.txt\nfile3.txt\nfile4.txt\nfile5.txt" | \
    xargs -P 4 -I {} bash -c 'process_file "{}"'

# Alternative: parallel (GNU parallel tool, if installed)
# seq 1 10 | parallel -j 4 'echo Processing {}; sleep 1'
```

### Parallel Processing Template

```bash
#!/bin/bash

# Generic parallel processor
parallel_process() {
    local max_parallel=$1
    local processor_func=$2
    shift 2
    local items=("$@")

    local count=0
    local pids=()

    for item in "${items[@]}"; do
        # Limit concurrency
        while [ "${#pids[@]}" -ge "$max_parallel" ]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}"
                    echo "Job for '$item' completed (PID: ${pids[$i]}, status: $?)"
                    unset 'pids[$i]'
                fi
            done
            pids=("${pids[@]}")
            sleep 0.1
        done

        # Start new job
        $processor_func "$item" &
        pids+=($!)
        ((count++))
    done

    # Wait for all remaining
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "Processed $count items"
}

# Example processor function
my_processor() {
    local item=$1
    sleep $((RANDOM % 3 + 1))
    echo "Processed: $item"
}

export -f my_processor

# Process 20 items with max 5 parallel
items=($(seq 1 20))
parallel_process 5 my_processor "${items[@]}"
```

## 3. Subshells

Subshells create isolated execution environments with their own variable scope.

### Subshell Syntax

```bash
#!/bin/bash

# Parentheses create a subshell
VAR="outer"

(
    VAR="inner"
    echo "Inside subshell: $VAR"
)

echo "Outside subshell: $VAR"  # Still "outer"

# Command substitution also creates subshell
result=$(
    VAR="command substitution"
    echo "$VAR"
)
echo "Result: $result"
echo "VAR is still: $VAR"  # Still "outer"
```

### Subshell vs Command Grouping

```bash
#!/bin/bash

# Subshell ( ) - separate process, isolated scope
VAR="original"
( VAR="subshell"; cd /tmp; pwd )
echo "VAR: $VAR"       # "original"
echo "PWD: $PWD"       # unchanged

# Command grouping { } - same process, shared scope
VAR="original"
{ VAR="grouped"; echo "In group: $VAR"; }
echo "VAR: $VAR"       # "grouped"

# Note: { } requires spaces and semicolon/newline before }
```

### Practical Subshell Uses

```bash
#!/bin/bash

# 1. Temporary directory changes
(cd /tmp && ls -la)  # Returns to original directory after

# 2. Temporary environment changes
(
    export PATH="/custom/path:$PATH"
    export CUSTOM_VAR="value"
    ./my_program  # Uses modified environment
)
# Environment restored here

# 3. Grouping for redirection
(
    echo "Log entry 1"
    echo "Log entry 2"
    echo "Log entry 3"
) >> logfile.txt

# 4. Background job isolation
for i in {1..3}; do
    (
        # Each iteration has isolated environment
        ITERATION=$i
        sleep 1
        echo "Iteration $ITERATION complete"
    ) &
done
wait

# 5. Pipeline with grouped commands
(echo "line 1"; echo "line 2"; echo "line 3") | grep "line 2"
```

### Variable Scope Implications

```bash
#!/bin/bash

# Variables modified in subshells don't persist
counter=0

while read line; do
    ((counter++))  # This won't work if while is in subshell!
done < <(seq 1 10)

echo "Counter: $counter"  # 10 (process substitution avoids subshell for while)

# This creates a subshell (pipe on while)
counter=0
seq 1 10 | while read line; do
    ((counter++))
done
echo "Counter: $counter"  # 0 (subshell isolated the changes)

# Solution 1: Use process substitution (shown above)
# Solution 2: Use here string or redirection
# Solution 3: Use a file descriptor
```

## 4. Signals

Signals are software interrupts that notify processes of events.

### Common Signals

| Signal | Number | Description | Default Action |
|--------|--------|-------------|----------------|
| SIGHUP | 1 | Hangup (terminal closed) | Terminate |
| SIGINT | 2 | Interrupt (Ctrl+C) | Terminate |
| SIGQUIT | 3 | Quit (Ctrl+\) | Terminate + core dump |
| SIGKILL | 9 | Kill (cannot be caught) | Terminate |
| SIGTERM | 15 | Termination request | Terminate |
| SIGSTOP | 19 | Stop (cannot be caught) | Stop |
| SIGCONT | 18 | Continue if stopped | Continue |
| SIGUSR1 | 10 | User-defined signal 1 | Terminate |
| SIGUSR2 | 12 | User-defined signal 2 | Terminate |
| SIGPIPE | 13 | Broken pipe | Terminate |
| SIGCHLD | 17 | Child process changed | Ignore |
| SIGALRM | 14 | Timer expired | Terminate |

### Sending Signals

```bash
#!/bin/bash

# Start a background process
sleep 100 &
PID=$!

# Send signals to process
kill -SIGTERM $PID  # Polite termination request
# kill -15 $PID     # Same, using number

# kill -SIGKILL $PID  # Force kill (cannot be caught)
# kill -9 $PID        # Same, using number

# Check if process is running (signal 0)
if kill -0 $PID 2>/dev/null; then
    echo "Process $PID is running"
else
    echo "Process $PID is not running"
fi

# Send signal to process group
# kill -TERM -$$  # Kills entire process group
```

### Listing Signals

```bash
#!/bin/bash

# List all signals
kill -l

# Get signal name from number
kill -l 9   # KILL

# Get signal number from name
kill -l TERM  # 15
```

## 5. trap Command

The `trap` command allows scripts to catch and handle signals.

### Basic trap Syntax

```bash
#!/bin/bash

# Trap syntax: trap 'commands' SIGNAL [SIGNAL...]

# Catch Ctrl+C (SIGINT)
trap 'echo "Caught SIGINT (Ctrl+C)! Exiting..."; exit 1' INT

echo "Press Ctrl+C to trigger the trap..."
sleep 30
echo "Completed normally"
```

### Trapping Multiple Signals

```bash
#!/bin/bash

cleanup() {
    echo "Cleanup function called by signal: $1"
    # Perform cleanup here
    exit 0
}

# Trap multiple signals
trap 'cleanup SIGINT' INT
trap 'cleanup SIGTERM' TERM
trap 'cleanup SIGHUP' HUP

echo "Script running (PID: $$)..."
echo "Try: kill -TERM $$"
while true; do
    sleep 1
done
```

### Ignoring Signals

```bash
#!/bin/bash

# Ignore SIGINT (empty string)
trap '' INT

echo "Try Ctrl+C - it won't work!"
sleep 5

# Reset to default behavior
trap - INT

echo "Now Ctrl+C will work again"
sleep 5
```

### Trap on EXIT

```bash
#!/bin/bash

# EXIT pseudo-signal: triggered on script exit (any reason)
trap 'echo "Script exiting..."' EXIT

echo "Starting script"
sleep 2
echo "Ending script"

# EXIT trap runs here, regardless of how script exits
```

### Debugging with trap

```bash
#!/bin/bash

# DEBUG pseudo-signal: executed before each command
trap 'echo "Executing: $BASH_COMMAND"' DEBUG

echo "First command"
x=10
echo "x is $x"
((x++))
echo "x is now $x"

trap - DEBUG  # Disable DEBUG trap
```

## 6. Cleanup Patterns

Proper cleanup ensures scripts don't leave behind temporary files, lock files, or orphaned processes.

### Temporary File Cleanup

```bash
#!/bin/bash

# Create temp file
TMPFILE=$(mktemp) || exit 1

# Ensure cleanup on exit
trap 'rm -f "$TMPFILE"' EXIT

echo "Using temp file: $TMPFILE"

# Use temp file
echo "data" > "$TMPFILE"
cat "$TMPFILE"

# Cleanup happens automatically on exit
```

### Comprehensive Cleanup

```bash
#!/bin/bash

# Cleanup function
cleanup() {
    local exit_code=$?

    echo "Performing cleanup..."

    # Remove temp files
    [ -n "$TMPFILE" ] && [ -f "$TMPFILE" ] && rm -f "$TMPFILE"
    [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ] && rm -rf "$TMPDIR"

    # Release lock file
    [ -n "$LOCKFILE" ] && [ -f "$LOCKFILE" ] && rm -f "$LOCKFILE"

    # Kill child processes
    [ -n "$WORKER_PID" ] && kill "$WORKER_PID" 2>/dev/null

    # Restore state
    [ -n "$ORIGINAL_DIR" ] && cd "$ORIGINAL_DIR"

    echo "Cleanup complete"
    exit "$exit_code"
}

# Set trap
trap cleanup EXIT INT TERM

# Remember original directory
ORIGINAL_DIR=$PWD

# Create temp resources
TMPFILE=$(mktemp)
TMPDIR=$(mktemp -d)
LOCKFILE="/tmp/myscript.lock"

echo "Resources created:"
echo "  TMPFILE: $TMPFILE"
echo "  TMPDIR: $TMPDIR"
echo "  LOCKFILE: $LOCKFILE"

# Simulate work
echo "Working..."
sleep 2

# Cleanup happens automatically
```

### Lock File Pattern

```bash
#!/bin/bash

LOCKFILE="/var/lock/myscript.lock"

# Acquire lock
acquire_lock() {
    if [ -e "$LOCKFILE" ]; then
        echo "Another instance is running (lock file exists)"
        exit 1
    fi

    # Create lock file with our PID
    echo $$ > "$LOCKFILE"

    # Ensure cleanup
    trap 'rm -f "$LOCKFILE"; exit' EXIT INT TERM
}

# Alternative: atomic lock with mkdir
acquire_lock_atomic() {
    local lockdir="/var/lock/myscript.lock.d"

    if mkdir "$lockdir" 2>/dev/null; then
        trap 'rmdir "$lockdir"; exit' EXIT INT TERM
        return 0
    else
        echo "Another instance is running"
        return 1
    fi
}

acquire_lock

echo "Lock acquired, doing work..."
sleep 5
echo "Done"
```

### Idempotent Cleanup

```bash
#!/bin/bash

# Cleanup that can be called multiple times safely
cleanup() {
    # Use flags to track what's been cleaned
    [ -n "$CLEANUP_DONE" ] && return
    CLEANUP_DONE=1

    echo "Running cleanup..."

    # Check before removing
    if [ -n "$TMPFILE" ] && [ -f "$TMPFILE" ]; then
        rm -f "$TMPFILE"
        echo "Removed $TMPFILE"
    fi

    # Kill process only if running
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill "$CHILD_PID"
        wait "$CHILD_PID" 2>/dev/null
        echo "Killed process $CHILD_PID"
    fi
}

trap cleanup EXIT INT TERM

TMPFILE=$(mktemp)
sleep 100 &
CHILD_PID=$!

echo "Press Ctrl+C to test cleanup..."
sleep 30
```

## 7. Coprocesses (coproc)

Bash 4.0+ supports coprocesses for bidirectional communication.

### Basic Coprocess

```bash
#!/bin/bash

# Start a coprocess
coproc BC { bc; }

# Write to coprocess (stdin)
echo "5 + 3" >&"${BC[1]}"

# Read from coprocess (stdout)
read -u "${BC[0]}" result
echo "Result: $result"

# Close coprocess
eval "exec ${BC[1]}>&-"
wait $BC_PID
```

### Named Coprocess

```bash
#!/bin/bash

# Named coprocess
coproc CALC { bc -l; }

# Function to calculate
calculate() {
    echo "$1" >&"${CALC[1]}"
    read -u "${CALC[0]}" result
    echo "$result"
}

# Use the calculator
echo "sqrt(16) = $(calculate 'sqrt(16)')"
echo "10 / 3 = $(calculate '10 / 3')"
echo "e(1) = $(calculate 'e(1)')"

# Cleanup
eval "exec ${CALC[1]}>&-"
wait $CALC_PID
```

### Interactive Coprocess Example

```bash
#!/bin/bash

# Start a shell coprocess
coproc SHELL { bash; }

# Function to execute command in coprocess
exec_in_coproc() {
    local cmd=$1
    echo "$cmd" >&"${SHELL[1]}"
    echo "echo '<<<END>>>'" >&"${SHELL[1]}"

    while read -u "${SHELL[0]}" line; do
        [ "$line" = "<<<END>>>" ] && break
        echo "$line"
    done
}

# Execute commands
echo "Current directory:"
exec_in_coproc "pwd"

echo -e "\nFiles:"
exec_in_coproc "ls -1"

echo -e "\nEnvironment variable:"
exec_in_coproc "echo \$HOME"

# Cleanup
echo "exit" >&"${SHELL[1]}"
wait $SHELL_PID
```

### Coprocess with Error Handling

```bash
#!/bin/bash

# Start coprocess with error handling
start_coproc() {
    if ! coproc WORKER { python3 -u -c '
import sys
while True:
    try:
        line = input()
        if line == "QUIT":
            break
        # Process line
        print(f"Processed: {line}")
        sys.stdout.flush()
    except EOFError:
        break
'; }; then
        echo "Failed to start coprocess"
        return 1
    fi

    # Setup cleanup
    trap 'echo "QUIT" >&"${WORKER[1]}" 2>/dev/null; wait $WORKER_PID 2>/dev/null' EXIT
}

# Use coprocess
send_to_worker() {
    echo "$1" >&"${WORKER[1]}"
    read -u "${WORKER[0]}" -t 5 response || {
        echo "Timeout or error reading from worker"
        return 1
    }
    echo "$response"
}

start_coproc

send_to_worker "task1"
send_to_worker "task2"
send_to_worker "task3"
```

## 8. Process Priority

Control CPU and I/O priority of processes.

### nice Command

```bash
#!/bin/bash

# Nice values range from -20 (highest priority) to 19 (lowest)
# Default is 0

# Run with lower priority (nice value 10)
nice -n 10 ./cpu-intensive-script.sh

# Run with higher priority (requires root for negative values)
# sudo nice -n -10 ./important-script.sh

# Check current nice value
echo "Current nice value: $(nice)"
```

### renice Command

```bash
#!/bin/bash

# Start a process
./my-script.sh &
PID=$!

# Change priority of running process
renice -n 15 -p $PID

# Renice all processes of a user
# sudo renice -n 10 -u username

# Renice all processes in a group
# sudo renice -n 5 -g groupname
```

### ionice Command

```bash
#!/bin/bash

# I/O scheduling classes:
# 0 - None (default)
# 1 - Real-time (highest priority, requires root)
# 2 - Best-effort (default)
# 3 - Idle (only when no other I/O)

# Run with idle I/O priority
ionice -c 3 ./disk-intensive-script.sh

# Run with best-effort, priority 4 (0-7, lower is higher priority)
ionice -c 2 -n 4 ./my-script.sh

# Change I/O priority of running process
ionice -c 3 -p $PID
```

### Combined Priority Example

```bash
#!/bin/bash

# Run CPU and I/O intensive task with low priority
run_low_priority() {
    local cmd=$1

    # Start with nice
    nice -n 19 bash -c "$cmd" &
    local pid=$!

    # Set I/O priority to idle
    ionice -c 3 -p $pid

    echo "Started low priority process: $pid"
    echo "Nice: $(ps -o nice= -p $pid)"
    echo "I/O class: $(ionice -p $pid)"

    wait $pid
}

# Example usage
run_low_priority "find / -name '*.log' 2>/dev/null | xargs gzip"
```

### Priority Management Script

```bash
#!/bin/bash

# Manage process priority
manage_priority() {
    local pid=$1
    local cpu_priority=$2  # -20 to 19
    local io_class=$3      # 0-3
    local io_priority=$4   # 0-7

    echo "Managing priority for PID $pid"

    # Set CPU priority
    if [ -n "$cpu_priority" ]; then
        if renice -n "$cpu_priority" -p "$pid" >/dev/null 2>&1; then
            echo "  CPU priority set to $cpu_priority"
        else
            echo "  Failed to set CPU priority (may need sudo)"
        fi
    fi

    # Set I/O priority
    if [ -n "$io_class" ]; then
        local ionice_cmd="ionice -c $io_class"
        [ -n "$io_priority" ] && ionice_cmd="$ionice_cmd -n $io_priority"

        if $ionice_cmd -p "$pid" >/dev/null 2>&1; then
            echo "  I/O priority set to class $io_class"
        else
            echo "  Failed to set I/O priority (may need sudo)"
        fi
    fi

    # Show current priorities
    echo "  Current nice value: $(ps -o nice= -p $pid)"
    echo "  Current I/O: $(ionice -p $pid | head -1)"
}

# Example: run backup with low priority
echo "Starting backup..."
./backup.sh &
BACKUP_PID=$!

manage_priority $BACKUP_PID 19 3
wait $BACKUP_PID
echo "Backup complete"
```

## Practice Problems

### Problem 1: Parallel File Processor

Write a script that processes multiple files in parallel with a configurable maximum number of concurrent jobs. Each file should be processed by a function that simulates work (sleep), and the script should report when each file starts and completes processing.

Requirements:
- Accept a directory path and max concurrent jobs as arguments
- Process all `.txt` files in the directory
- Track and report total processing time
- Handle errors gracefully

### Problem 2: Signal-Safe Download Manager

Create a download manager script that:
- Downloads multiple URLs in parallel
- Saves progress to a state file
- Can be interrupted with SIGINT (Ctrl+C) and resume later
- Cleans up partial downloads on SIGTERM
- Uses trap to handle all cleanup properly

### Problem 3: Background Job Monitor

Implement a job monitoring system that:
- Starts multiple background jobs
- Monitors their status every second
- Reports when each job completes
- Shows a progress indicator
- Limits concurrent jobs to a maximum (e.g., 3)
- Handles job failures and retries failed jobs once

### Problem 4: Coprocess Calculator Service

Build a calculator service using coprocesses:
- Start a `bc` coprocess
- Provide a command-line interface to enter expressions
- Support history (show last 10 calculations)
- Handle errors (invalid expressions)
- Implement a proper shutdown sequence

### Problem 5: Priority-Based Task Scheduler

Create a task scheduler that:
- Accepts tasks with priority levels (high, medium, low)
- Runs high-priority tasks with nice value 0
- Runs medium-priority tasks with nice value 10
- Runs low-priority tasks with nice value 19 and I/O class idle
- Reports the status of all running tasks
- Limits total concurrent tasks to 5
- Implements proper cleanup on exit

---

**Previous**: [Regular Expressions in Bash](./08_Regex_in_Bash.md) | **Next**: [Error Handling and Debugging](./10_Error_Handling.md)
