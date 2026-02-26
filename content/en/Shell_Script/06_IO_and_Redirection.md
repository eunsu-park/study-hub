# Lesson 06: I/O and Redirection

**Difficulty**: ⭐⭐⭐

**Previous**: [Functions and Libraries](./05_Functions_and_Libraries.md) | **Next**: [String Processing and Text Manipulation](./07_String_Processing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how file descriptors work and create custom FDs for multi-stream I/O
2. Apply advanced redirection techniques to separate, merge, swap, and save/restore stdout and stderr
3. Write here documents and here strings for inline data and template generation
4. Use process substitution to avoid subshell variable scope issues in pipelines
5. Implement producer-consumer communication using named pipes (FIFOs)
6. Identify common pipe pitfalls including subshell scope loss and PIPESTATUS checking
7. Write atomic file updates, file-locking routines, and multi-destination logging patterns

---

I/O redirection is what makes shell scripting powerful for data pipelines, logging, and process coordination. Beyond simple `>` and `|`, bash provides file descriptors, process substitution, here documents, and named pipes that let you build sophisticated data flows without intermediate files. You need these skills whenever you write log handlers, parse multi-stream command output, or build concurrent producer-consumer workflows.

## 1. File Descriptors

File descriptors (FDs) are integers that reference open files or I/O streams. Understanding them is fundamental to mastering I/O redirection in bash.

### 1.1 Standard File Descriptors

Every process has three standard file descriptors:

| FD | Name | Purpose | Default |
|----|------|---------|---------|
| 0 | stdin | Standard input | Keyboard |
| 1 | stdout | Standard output | Terminal |
| 2 | stderr | Standard error | Terminal |

```bash
#!/bin/bash

# Read from stdin (FD 0)
read -p "Enter your name: " name
echo "Hello, $name"

# Write to stdout (FD 1)
echo "This goes to stdout" >&1  # Explicit (same as just echo)

# Write to stderr (FD 2)
echo "This is an error message" >&2
```

### 1.2 Custom File Descriptors

You can create custom file descriptors (3-9 are commonly used):

```bash
#!/bin/bash

# Open file for reading on FD 3
exec 3< input.txt

# Read from FD 3
while read -u 3 line; do
    echo "Line: $line"
done

# Close FD 3
exec 3<&-

# Open file for writing on FD 4
exec 4> output.txt

# Write to FD 4
echo "First line" >&4
echo "Second line" >&4

# Close FD 4
exec 4>&-
```

### 1.3 Opening File Descriptors for Read/Write

```bash
#!/bin/bash

# Open file for both reading and writing on FD 5
exec 5<> datafile.txt

# Read current content
while read -u 5 line; do
    echo "Read: $line"
done

# Write new content (appends)
echo "New data" >&5

# Close FD 5
exec 5>&-
```

### 1.4 Duplicating File Descriptors

```bash
#!/bin/bash

# Duplicate stdout (FD 1) to FD 3
exec 3>&1

# Now redirect stdout to a file
exec 1> output.log

# This goes to output.log
echo "Logging to file"

# This still goes to terminal (via FD 3)
echo "Direct to terminal" >&3

# Restore stdout from FD 3
exec 1>&3

# Close FD 3
exec 3>&-

# Now back to terminal
echo "Back to normal stdout"
```

### 1.5 File Descriptor Inspection

```bash
#!/bin/bash

# View file descriptors for current shell
ls -l /dev/fd/
# or
ls -l /proc/self/fd/

# Check if FD is open
if [[ -e /dev/fd/3 ]]; then
    echo "FD 3 is open"
else
    echo "FD 3 is closed"
fi

# Get information about FD
exec 5> myfile.txt
readlink /proc/self/fd/5  # Shows the file path
exec 5>&-
```

## 2. Advanced Redirection

Beyond basic `>` and `<`, bash offers powerful redirection operators.

### 2.1 Redirecting Stderr Separately

```bash
#!/bin/bash

# Redirect stdout to file1, stderr to file2
command > stdout.log 2> stderr.log

# Example: compile C program
gcc program.c -o program > compile_output.txt 2> compile_errors.txt

# Check if compilation had errors
if [[ -s compile_errors.txt ]]; then
    echo "Compilation failed:"
    cat compile_errors.txt
else
    echo "Compilation successful!"
fi
```

### 2.2 Merging Stdout and Stderr

```bash
#!/bin/bash

# Method 1: Redirect stderr to stdout
command > output.log 2>&1

# Method 2: Shorthand (Bash 4+)
command &> output.log

# Method 3: Append both
command >> output.log 2>&1

# Example: run test suite
./run_tests.sh &> test_results.log

# This is WRONG (order matters):
command 2>&1 > output.log  # stderr still goes to terminal!
# Correct:
command > output.log 2>&1  # stderr follows stdout to file
```

### 2.3 Discarding Output

```bash
#!/bin/bash

# Discard stdout
command > /dev/null

# Discard stderr
command 2> /dev/null

# Discard both
command &> /dev/null

# Example: silent operation
if some_command &> /dev/null; then
    echo "Command succeeded (silently)"
fi

# Keep stderr, discard stdout
command > /dev/null

# Example: check if command exists
if command -v python3 > /dev/null 2>&1; then
    echo "python3 is installed"
fi
```

### 2.4 Swapping Stdout and Stderr

```bash
#!/bin/bash

# Swap stdout and stderr
command 3>&1 1>&2 2>&3 3>&-

# Explanation:
# 3>&1  - Save stdout to FD 3
# 1>&2  - Redirect stdout to stderr
# 2>&3  - Redirect stderr to FD 3 (original stdout)
# 3>&-  - Close FD 3

# Practical example: error messages to stdout, normal output to stderr
swap_outputs() {
    "$@" 3>&1 1>&2 2>&3 3>&-
}

# Now errors appear on stdout (can be captured)
errors=$(swap_outputs some_command)
```

### 2.5 Saving and Restoring File Descriptors

```bash
#!/bin/bash

# Save original stdout and stderr
exec 3>&1 4>&2

# Redirect stdout and stderr to files
exec 1> output.log 2> error.log

# Commands here write to log files
echo "This goes to output.log"
echo "This is an error" >&2

# Restore original stdout and stderr
exec 1>&3 2>&4

# Close backup FDs
exec 3>&- 4>&-

# Now back to terminal
echo "Back to terminal"
```

### 2.6 Appending vs Truncating

```bash
#!/bin/bash

# Truncate file (overwrite)
echo "New content" > file.txt

# Append to file
echo "Additional content" >> file.txt

# Append stderr
command 2>> error.log

# Append both stdout and stderr
command &>> output.log
```

## 3. Here Documents and Here Strings

Here documents provide multi-line input to commands without creating temporary files.

### 3.1 Basic Here Document

```bash
#!/bin/bash

# Basic here document
cat <<EOF
This is a multi-line
here document.
It can contain variables: $HOME
And command substitution: $(date)
EOF

# With indentation (<<- removes leading tabs, not spaces)
cat <<-EOF
	This is indented with tabs
	The tabs will be removed
	But the text stays aligned
EOF
```

### 3.2 Here Document Without Variable Expansion

```bash
#!/bin/bash

# Quote the delimiter to prevent expansion
cat <<'EOF'
Variables are literal: $HOME
Command substitution is literal: $(date)
This is useful for generating scripts or code.
EOF

# Example: generate a bash script
cat <<'SCRIPT' > myscript.sh
#!/bin/bash
echo "Hello from generated script"
echo "Current directory: $PWD"
SCRIPT

chmod +x myscript.sh
```

### 3.3 Here Document to Variables

```bash
#!/bin/bash

# Assign here document to variable
read -r -d '' sql_query <<EOF
SELECT u.name, u.email, o.order_id
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.status = 'pending'
ORDER BY o.created_at DESC
LIMIT 10;
EOF

echo "Executing query:"
echo "$sql_query"

# Alternative method (using command substitution)
json_data=$(cat <<EOF
{
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30,
    "roles": ["admin", "user"]
}
EOF
)

echo "$json_data"
```

### 3.4 Here Document with Command Input

```bash
#!/bin/bash

# Send multi-line input to a command
mysql -u root -p <<SQL
USE mydb;
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO users (username, email) VALUES ('alice', 'alice@example.com');
SQL

# Python script execution
python3 <<PYTHON
import sys
import json

data = {
    'message': 'Hello from Python',
    'version': sys.version
}

print(json.dumps(data, indent=2))
PYTHON
```

### 3.5 Here Strings

```bash
#!/bin/bash

# Here string: single-line input
grep "pattern" <<< "This is a test pattern string"

# Useful for piping variables
while read -r word; do
    echo "Word: $word"
done <<< "one two three four five"

# Example: parse CSV line
IFS=',' read -r name age city <<< "John,30,NYC"
echo "Name: $name, Age: $age, City: $city"

# Base64 encode a string
encoded=$(base64 <<< "Secret message")
echo "Encoded: $encoded"

# Decode it back
decoded=$(base64 -d <<< "$encoded")
echo "Decoded: $decoded"
```

### 3.6 Practical Template Generation

```bash
#!/bin/bash

generate_html() {
    local title=$1
    local content=$2

    cat <<HTML
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>$title</title>
</head>
<body>
    <h1>$title</h1>
    <p>$content</p>
    <footer>Generated on $(date)</footer>
</body>
</html>
HTML
}

# Generate HTML page
generate_html "My Page" "Welcome to my website!" > index.html

# Generate configuration file
generate_config() {
    local host=$1
    local port=$2

    cat <<CONFIG > app.conf
# Application Configuration
# Generated: $(date)

[server]
host = $host
port = $port
workers = 4

[database]
host = localhost
port = 5432
name = myapp

[logging]
level = INFO
file = /var/log/myapp.log
CONFIG
}

generate_config "0.0.0.0" "8080"
```

## 4. Process Substitution

Process substitution creates temporary named pipes for command output, allowing commands to be used where files are expected.

### 4.1 Input Process Substitution

```bash
#!/bin/bash

# Compare output of two commands
diff <(ls dir1) <(ls dir2)

# More complex example: compare sorted lists
diff <(sort file1.txt) <(sort file2.txt)

# Compare running processes on two systems
diff <(ssh server1 ps aux | sort) <(ssh server2 ps aux | sort)

# Example: find common lines in two command outputs
comm -12 <(sort list1.txt) <(sort list2.txt)
```

### 4.2 Output Process Substitution

```bash
#!/bin/bash

# Write to multiple files simultaneously
tee >(grep "ERROR" > errors.log) \
    >(grep "WARN" > warnings.log) \
    >(grep "INFO" > info.log) \
    < application.log > /dev/null

# Example: split log by severity
process_logs() {
    local logfile=$1

    cat "$logfile" | tee \
        >(grep "ERROR" > errors.log) \
        >(grep "WARN" > warnings.log) \
        > all.log
}
```

### 4.3 Avoiding Subshell Variable Scope Issues

```bash
#!/bin/bash

# PROBLEM: Variables in pipeline are in subshell
count=0
cat file.txt | while read line; do
    ((count++))
done
echo "Lines: $count"  # Output: 0 (variable not modified!)

# SOLUTION 1: Process substitution
count=0
while read line; do
    ((count++))
done < <(cat file.txt)
echo "Lines: $count"  # Correct count

# SOLUTION 2: Use here string with command substitution (for small files)
count=0
while read line; do
    ((count++))
done <<< "$(cat file.txt)"
echo "Lines: $count"  # Correct count
```

### 4.4 Multiple Input Streams

```bash
#!/bin/bash

# Read from multiple files in parallel
paste <(cut -d',' -f1 file1.csv) \
      <(cut -d',' -f2 file2.csv) \
      <(cut -d',' -f3 file3.csv)

# Example: merge data from multiple sources
while read -u 3 name && read -u 4 age && read -u 5 city; do
    echo "$name is $age years old and lives in $city"
done 3< <(cut -d',' -f1 data.csv) \
     4< <(cut -d',' -f2 data.csv) \
     5< <(cut -d',' -f3 data.csv)
```

### 4.5 Practical Examples

```bash
#!/bin/bash

# Example 1: Find files modified in last 24 hours that contain pattern
grep "TODO" <(find . -type f -mtime -1 -exec cat {} \;)

# Example 2: Monitor log file and send alerts
while read line; do
    if [[ $line == *"CRITICAL"* ]]; then
        echo "Alert: $line" | mail -s "Critical Error" admin@example.com
    fi
done < <(tail -f /var/log/app.log)

# Example 3: Process compressed file without extracting
while read line; do
    echo "Processing: $line"
done < <(gunzip -c data.txt.gz)

# Example 4: Create temporary file list for processing
tar czf backup.tar.gz -T <(find /data -type f -mtime -7)
```

## 5. Named Pipes (FIFOs)

Named pipes allow inter-process communication through the filesystem.

### 5.1 Creating and Using FIFOs

```bash
#!/bin/bash

# Create named pipe
mkfifo mypipe

# Producer (background process)
{
    for i in {1..10}; do
        echo "Message $i"
        sleep 1
    done > mypipe
} &

# Consumer
while read line; do
    echo "Received: $line"
done < mypipe

# Cleanup
rm mypipe
```

### 5.2 Producer-Consumer Pattern

```bash
#!/bin/bash

PIPE="/tmp/data_pipe_$$"

# Create pipe and set trap for cleanup
mkfifo "$PIPE"
trap "rm -f '$PIPE'" EXIT

# Producer: generate data
producer() {
    local pipe=$1
    echo "Producer starting..."

    for i in {1..100}; do
        echo "Data item $i: $(date +%s)"
        sleep 0.1
    done > "$pipe"

    echo "Producer finished"
}

# Consumer: process data
consumer() {
    local pipe=$1
    echo "Consumer starting..."

    local count=0
    while read line; do
        ((count++))
        # Process data (simulate work)
        [[ $((count % 10)) -eq 0 ]] && echo "Processed $count items"
    done < "$pipe"

    echo "Consumer finished: $count items processed"
}

# Run producer in background
producer "$PIPE" &
producer_pid=$!

# Run consumer in foreground
consumer "$PIPE"

# Wait for producer to finish
wait $producer_pid
```

### 5.3 Bidirectional Communication

```bash
#!/bin/bash

REQUEST_PIPE="/tmp/request_$$"
RESPONSE_PIPE="/tmp/response_$$"

# Create pipes
mkfifo "$REQUEST_PIPE" "$RESPONSE_PIPE"
trap "rm -f '$REQUEST_PIPE' '$RESPONSE_PIPE'" EXIT

# Server process
server() {
    echo "Server started"

    while true; do
        # Read request
        read request < "$REQUEST_PIPE"

        # Process request
        case $request in
            "PING")
                echo "PONG" > "$RESPONSE_PIPE"
                ;;
            "TIME")
                date > "$RESPONSE_PIPE"
                ;;
            "QUIT")
                echo "BYE" > "$RESPONSE_PIPE"
                break
                ;;
            *)
                echo "ERROR: Unknown command" > "$RESPONSE_PIPE"
                ;;
        esac
    done

    echo "Server stopped"
}

# Client function
client() {
    local command=$1

    # Send request
    echo "$command" > "$REQUEST_PIPE"

    # Read response
    read response < "$RESPONSE_PIPE"
    echo "Response: $response"
}

# Start server in background
server &
server_pid=$!

sleep 1  # Give server time to start

# Send requests
client "PING"
client "TIME"
client "QUIT"

# Wait for server
wait $server_pid
```

### 5.4 When to Use FIFOs vs Process Substitution

| Feature | FIFO | Process Substitution |
|---------|------|---------------------|
| Persistence | Yes (until deleted) | No (automatic cleanup) |
| Multiple readers/writers | Yes | No |
| Explicit synchronization | Yes | No |
| Use in background | Easy | Complex |
| Cleanup required | Manual | Automatic |
| Best for | Long-running IPC | One-time operations |

```bash
#!/bin/bash

# Use process substitution for one-time comparison
diff <(command1) <(command2)

# Use FIFO for persistent communication
mkfifo /tmp/logpipe
tail -f /var/log/app.log > /tmp/logpipe &
while read line; do
    process_log_line "$line"
done < /tmp/logpipe
```

## 6. Pipe Pitfalls and Solutions

### 6.1 Subshell Variable Scope Loss

```bash
#!/bin/bash

# PROBLEM: Last command in pipeline runs in subshell
total=0
cat numbers.txt | while read num; do
    ((total += num))
done
echo "Total: $total"  # Output: 0 (not modified!)

# SOLUTION 1: Process substitution
total=0
while read num; do
    ((total += num))
done < <(cat numbers.txt)
echo "Total: $total"  # Correct

# SOLUTION 2: Use lastpipe (Bash 4.2+, only in scripts)
shopt -s lastpipe
total=0
cat numbers.txt | while read num; do
    ((total += num))
done
echo "Total: $total"  # Correct

# SOLUTION 3: Temporary file
tmpfile=$(mktemp)
cat numbers.txt > "$tmpfile"
total=0
while read num; do
    ((total += num))
done < "$tmpfile"
rm "$tmpfile"
echo "Total: $total"  # Correct
```

### 6.2 PIPESTATUS Array

```bash
#!/bin/bash

# Check exit status of all pipeline commands
command1 | command2 | command3

# PIPESTATUS contains exit codes of all commands
echo "Exit codes: ${PIPESTATUS[@]}"

# Example: detect failure anywhere in pipeline
false | true | true
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "First command failed"
fi

# Practical example: database pipeline
{
    mysql -u root -p mydb -e "SELECT * FROM users" | \
    grep "active" | \
    sort -k2
} 2>/dev/null

pipeline_status=("${PIPESTATUS[@]}")
if [[ ${pipeline_status[0]} -ne 0 ]]; then
    echo "Database query failed"
elif [[ ${pipeline_status[1]} -ne 0 ]]; then
    echo "Grep failed"
elif [[ ${pipeline_status[2]} -ne 0 ]]; then
    echo "Sort failed"
else
    echo "Pipeline succeeded"
fi
```

### 6.3 Pipeline Error Handling

```bash
#!/bin/bash

# Enable pipefail: pipeline fails if any command fails
set -o pipefail

# Now the pipeline returns non-zero if any command fails
if command1 | command2 | command3; then
    echo "Pipeline succeeded"
else
    echo "Pipeline failed"
fi

# Practical example: safe data processing
set -euo pipefail  # Exit on error, undefined variable, or pipeline failure

process_data() {
    local input=$1
    local output=$2

    cat "$input" | \
        grep -v "^#" | \
        sort -u | \
        sed 's/foo/bar/g' \
        > "$output"

    # If any command fails, script exits
}

# Error handling with pipefail
set -o pipefail
if ! tar czf backup.tar.gz --exclude="*.tmp" -T <(find /data -type f); then
    echo "Backup failed" >&2
    exit 1
fi
```

### 6.4 Named Pipe Deadlock Prevention

```bash
#!/bin/bash

# PROBLEM: Deadlock if reader/writer not coordinated
mkfifo mypipe
echo "data" > mypipe  # BLOCKS forever (no reader)!

# SOLUTION 1: Open pipe for reading and writing
mkfifo mypipe
exec 3<> mypipe  # Open for read/write

echo "data" >&3  # Write
read line <&3    # Read
exec 3>&-        # Close

rm mypipe

# SOLUTION 2: Background processes with proper synchronization
mkfifo mypipe
trap "rm -f mypipe" EXIT

# Reader in background
cat < mypipe &
reader_pid=$!

# Writer
echo "data" > mypipe

# Wait for reader
wait $reader_pid
```

## 7. Practical I/O Patterns

### 7.1 Tee to Multiple Destinations

```bash
#!/bin/bash

# Basic tee: write to file and stdout
echo "Important message" | tee log.txt

# Multiple files
echo "Message" | tee file1.txt file2.txt file3.txt

# Append mode
echo "New entry" | tee -a logfile.txt

# Complex example: split processing
cat data.txt | tee \
    >(grep "ERROR" > errors.log) \
    >(grep "WARN" > warnings.log) \
    >(wc -l > linecount.txt) \
    | grep "INFO" > info.log
```

### 7.2 Logging to Console and File

```bash
#!/bin/bash

# Setup logging
LOGFILE="application.log"

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Log to both console and file
    echo "[$timestamp] [$level] $message" | tee -a "$LOGFILE"
}

# Usage
log "INFO" "Application started"
log "WARN" "Configuration file not found, using defaults"
log "ERROR" "Failed to connect to database"

# Alternative: redirect all output
exec > >(tee -a "$LOGFILE")
exec 2>&1

# Now all output goes to both console and file
echo "This appears in both places"
ls /nonexistent  # Error also logged
```

### 7.3 Reading and Writing Same File Safely

```bash
#!/bin/bash

# WRONG: This truncates the file before reading!
sort file.txt > file.txt  # file.txt becomes empty!

# SOLUTION 1: Use sponge (from moreutils)
sort file.txt | sponge file.txt

# SOLUTION 2: Use temporary file
sort file.txt > file.txt.tmp && mv file.txt.tmp file.txt

# SOLUTION 3: In-place edit with -i flag (if supported)
sed -i 's/foo/bar/g' file.txt

# Atomic file replacement
update_config() {
    local config_file=$1
    local tmpfile=$(mktemp)

    # Process file
    process_config < "$config_file" > "$tmpfile"

    # Atomic replacement
    mv "$tmpfile" "$config_file"
}
```

### 7.4 Atomic File Writes

```bash
#!/bin/bash

# Atomic write pattern: write to temp, then move
atomic_write() {
    local target_file=$1
    local content=$2

    local tmpfile=$(mktemp "${target_file}.XXXXXX")

    # Write to temporary file
    echo "$content" > "$tmpfile"

    # Verify write succeeded
    if [[ $? -eq 0 ]]; then
        # Atomic move (on same filesystem)
        mv "$tmpfile" "$target_file"
    else
        rm -f "$tmpfile"
        return 1
    fi
}

# Usage
atomic_write "config.json" '{"setting": "value"}'

# Complex example: update critical file
update_critical_file() {
    local file=$1
    local tmpfile=$(mktemp)

    # Set trap for cleanup
    trap "rm -f '$tmpfile'" RETURN

    # Generate new content
    if ! generate_content > "$tmpfile"; then
        echo "Error: Failed to generate content" >&2
        return 1
    fi

    # Validate new content
    if ! validate_content "$tmpfile"; then
        echo "Error: Content validation failed" >&2
        return 1
    fi

    # Set same permissions as original
    chmod --reference="$file" "$tmpfile" 2>/dev/null

    # Atomic replacement
    mv "$tmpfile" "$file"
}
```

### 7.5 File Locking for Safe Concurrent Access

```bash
#!/bin/bash

# Use flock for file locking
update_counter() {
    local counter_file="counter.txt"
    local lockfile="counter.lock"

    # Acquire exclusive lock (FD 200)
    {
        flock -x 200

        # Read current value
        local count=0
        [[ -f $counter_file ]] && count=$(cat "$counter_file")

        # Increment
        ((count++))

        # Write back
        echo "$count" > "$counter_file"

        echo "Counter updated to: $count"

    } 200>"$lockfile"
}

# Multiple processes can safely call this
for i in {1..10}; do
    update_counter &
done
wait

# Final value
echo "Final count: $(cat counter.txt)"

# Alternative: inline locking
{
    flock -x 200

    # Critical section
    echo "Exclusive access to resource"
    sleep 2

} 200>/tmp/mylock
```

### 7.6 Progress Indication with FIFOs

```bash
#!/bin/bash

# Create progress pipe
mkfifo /tmp/progress_$$
trap "rm -f /tmp/progress_$$" EXIT

# Progress monitor (background)
{
    while read percent message; do
        printf "\r[%-50s] %d%% %s" \
            "$(printf '#%.0s' $(seq 1 $((percent / 2))))" \
            "$percent" \
            "$message"
    done < /tmp/progress_$$
    echo
} &
monitor_pid=$!

# Worker process
{
    total=100
    for i in $(seq 1 $total); do
        # Simulate work
        sleep 0.05

        # Report progress
        percent=$((i * 100 / total))
        echo "$percent Processing item $i" > /tmp/progress_$$
    done
} &
worker_pid=$!

# Wait for completion
wait $worker_pid
wait $monitor_pid
```

## Practice Problems

### Problem 1: Multi-Target Logger
Create a logging system that:
- Accepts log level (DEBUG, INFO, WARN, ERROR) and message
- Writes all logs to `all.log`
- Writes ERROR logs to `error.log`
- Writes WARN and ERROR to `important.log`
- Displays ERROR and WARN on stderr, others on stdout
- Adds timestamp and hostname to each log entry
- Implements log rotation when file exceeds 10MB

### Problem 2: Pipeline Monitor
Write a script that:
- Runs a multi-stage pipeline (e.g., download | decompress | process | upload)
- Monitors the exit status of each stage using PIPESTATUS
- Logs progress to a file using process substitution
- Implements retry logic for failed stages
- Reports which stage failed and why
- Calculates total time and throughput

### Problem 3: FIFO-Based Queue System
Implement a simple job queue using named pipes:
- Create `job_submit` command that sends jobs to a queue
- Create `job_worker` that processes jobs from the queue
- Support multiple concurrent workers
- Implement job status tracking (pending, running, completed, failed)
- Handle worker crashes gracefully
- Provide `job_status` command to check queue state

### Problem 4: Configuration Validator
Build a tool that:
- Reads configuration file from stdin or file argument
- Validates syntax using a validation command
- If valid, atomically replaces the old config
- If invalid, shows errors on stderr and keeps old config
- Creates backup before replacement (keep last 5 backups)
- Logs all changes with timestamp
- Supports dry-run mode (validate without replacing)

### Problem 5: Stream Processor with FDs
Create a stream processing framework:
- Opens 3 input streams on FDs 3, 4, 5
- Merges streams with timestamps
- Filters based on regex pattern
- Splits output to different files based on content
- Maintains statistics (lines processed per stream, matches, errors)
- Uses process substitution for real-time monitoring
- Handles stream termination gracefully

**Previous**: [Functions and Libraries](./05_Functions_and_Libraries.md) | **Next**: [String Processing and Text Manipulation](./07_String_Processing.md)
